import torch
from torch.utils.data import DataLoader
import torcheval.metrics
import tqdm
import time
import torch.nn.functional as F

from metrics import StdDev, grad_norm, MacroMetric, RandomMetric, MostProbableMetric

def reduce_xor(x):
    """
    Naive fuzzy parity function.
    :param x: Tensor of probabilities.
    :return: The probability of the cumulative exclusive or of the input.
    """
    out = 1.0
    for i in range(x.size(0)):
        out = (out + x[i] - out * x[i]) * (1.0 - out * x[i])

    return out

def fuzzy_loss(pred, labels):
    """
    Loss from <https://proceedings.kr.org/2023/65/kr2023-0065-umili-et-al.pdf>
    :param pred: Predicted state probabilities.
    :param labels: Tensor of accepting states.
    :return:
    """
    accepting_loss = 1.0 - reduce_xor(pred[torch.nonzero(labels, as_tuple=True)])
    non_accepting_loss = 1.0 - torch.prod(pred[torch.nonzero(1 - labels, as_tuple=True)])

    return accepting_loss + non_accepting_loss



def train(net, classes, train_ds, val_ds, test_ds, opts):
    """
    Train the model on the provided datasets. If some events are triggered during training, it collects them on a set of flags.
    Generator returning tuples in the form ("epoch", epoch_metrics, tags) for each epoch. Phase is a string in ["start", "epoch", "end"]
    In order to correctly synchronize with the watchdog, it must yield two sentinels:
    ("start", None, None) signals the end of the initialization phase and the start of proper training.
    ("end", full_history, tags) signals the end of training, possibly providing additional tags (e.g., "Success" or "NaN loss").
    :param net: torch.nn.Module to train.
    :param classes: Nested dictionary of class values.
    :param train_ds: Train dataset.
    :param val_ds: Validation dataset.
    :param test_ds: Test dataset.
    :param opts: Dictionary of hyper-parameters.
    :return: Generator yielding tuples (phase, metrics evaluated for the current epoch, set of events triggered during the current epoch).
    """



    train_dl = DataLoader(train_ds, batch_size=opts["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=opts["batch_size"], shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=opts["batch_size"], shuffle=False)

    cls_size = [len(classes[k]) for k in sorted(classes.keys())]

    net.to(opts["device"])


    if opts["pretraining_lr"] > 0.0:
        optimizer = torch.optim.SGD(net.parameters(), lr=opts["pretraining_lr"])
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=-opts["pretraining_lr"])

    if opts["grad_clipping"] > 0.0:
        for p in net.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -opts["grad_clipping"], opts["grad_clipping"]))

    history = []
    ok = True
    return_tags = set()

    metrics = {
        s: {
            "label_acc": MacroMetric(len(net.backbone.keys()), torcheval.metrics.MulticlassAccuracy, opts["device"]),
            "const_acc": MacroMetric(len(net.propositions), torcheval.metrics.BinaryAccuracy, opts["device"]),
            "succ_acc": torcheval.metrics.MulticlassAccuracy(device=opts["device"]),
            "seq_acc": torcheval.metrics.BinaryAccuracy(device=opts["device"]),
        } for s in ["train", "val", "test"]
    }

    baselines = {k: {
            "rnd": {
                "succ_acc": RandomMetric(torcheval.metrics.MulticlassAccuracy, opts["device"]),
                "seq_acc": RandomMetric(torcheval.metrics.BinaryAccuracy, opts["device"]),
            },
            "mp": {
                "succ_acc": MostProbableMetric(torcheval.metrics.MulticlassAccuracy, opts["device"]),
                "seq_acc": MostProbableMetric(torcheval.metrics.BinaryAccuracy, opts["device"])
            }
        } for k in ["train", "val", "test"]
    }

    # Additional metrics.
    metrics["train"]["loss"] = torcheval.metrics.Mean(device=opts["device"])
    metrics["train"]["avg_grad_norm"] = torcheval.metrics.Mean(device=opts["device"])
    metrics["train"]["std_grad_norm"] = StdDev(device=opts["device"])

    yield ("start", None, None) # Sentinel to start the watchdog heartbeat.


    for e in tqdm.trange(opts["pretraining_epochs"] + opts["epochs"], position=0, desc="epoch", disable=opts["verbose"] < 1):
        # At the end of pre-training, reset the optimizer.
        if e == opts["pretraining_epochs"]:
            if opts["ablate_fc_after_pretraining"]:
                for v in net.backbone.values():
                    v.last_layer.reset_parameters()

            if opts["lr"] > 0.0:
                optimizer = torch.optim.SGD(net.parameters(), lr=opts["lr"])
            else:
                optimizer = torch.optim.Adam(net.parameters(), lr=-opts["lr"])


        for v in metrics.values():
            for k2, v2 in v.items():
                v2.reset()

        # To assess generalization, replace computed histograms for validation and test set with training set.
        for k in ["val", "test"]:
            for k2, v in baselines[k]["mp"].items():
                v.set_histogram(baselines["train"]["mp"][k2].get_histogram())

        net.train()

        start_time = time.time()
        has_been_updated = False
        for b, batch in (bar := tqdm.tqdm(enumerate(train_dl), position=1, desc="batch", leave=False, ncols=0, disable=opts["verbose"] < 2)):
            loss, ok, step_tags, updt = train_step(net, optimizer, batch, metrics, baselines, cls_size, e < opts["pretraining_epochs"], opts)

            has_been_updated |= updt
            return_tags.update(step_tags)

            bar.set_postfix_str(
                "{}Loss: {:.02f}, avg_Loss: {:.02f}, avg_dLoss: {:.02f}, std_dLoss: {:.02f}, Var_acc: {:.02f}, Con_acc: {:.02f}, Suc_acc: {:.02f}, Seq_acc: {:.02f}".format(
                    ("[Pre-training] " if e < opts["pretraining_epochs"] else ""), loss,
                    metrics["train"]["loss"].compute(), metrics["train"]["avg_grad_norm"].compute(),
                    metrics["train"]["std_grad_norm"].compute(), metrics["train"]["label_acc"].compute(),
                    metrics["train"]["const_acc"].compute(), metrics["train"]["succ_acc"].compute(),
                    metrics["train"]["seq_acc"].compute()))

            if e < opts["pretraining_epochs"] and opts["pretraining_batches"] > 0 and b >= opts["pretraining_batches"]:
                break

            # If there was an error, abort training.
            if not ok:
                break

        if not has_been_updated:
            ok = False
            return_tags.add("No gradient update")

        times = {"train": time.time() - start_time}
        with torch.no_grad():
            net.eval()
            for split, dl in {"val": val_dl, "test": test_dl}.items():
                start_time = time.time()
                for batch in (bar := tqdm.tqdm(dl, position=1, desc="batch", leave=False, ncols=0, disable=opts["verbose"] < 2)):
                    eval_step(net, batch, metrics, baselines, split, cls_size, opts)

                times[split] = time.time() - start_time

                bar.set_postfix_str(
                    "({}) Var_acc: {:.02f}, Con_acc: {:.02f}, Suc_acc: {:.02f}, Seq_acc: {:.02f}".format(
                        split,
                        metrics[split]["label_acc"].compute(),
                        metrics[split]["const_acc"].compute(),
                        metrics[split]["succ_acc"].compute(),
                        metrics[split]["seq_acc"].compute()))

        epoch_stats = {}
        for split in ["train", "val", "test"]:
            epoch_stats["{}/time".format(split)] = times[split]
            tmp = 0.0
            i = 0
            for k, v in metrics[split].items():
                epoch_stats["{}/{}".format(split, k)] = v.compute().item()
                if k.endswith("_acc"):
                    tmp += epoch_stats["{}/{}".format(split, k)]
                    i += 1
            epoch_stats["{}/avg_acc".format(split)] = tmp / max(1, i)

        # "Generalization" metrics: train-test and train/test for every recorded metric.
        for k in metrics["test"].keys():
            epoch_stats["delta/{}".format(k)] = epoch_stats["train/{}".format(k)] - epoch_stats["test/{}".format(k)]
            if epoch_stats["delta/{}".format(k)] > opts["overfitting_threshold"]:
                return_tags.add("Overfitting {} (learning)".format(k))

            if epoch_stats["test/{}".format(k)] != 0:
                epoch_stats["ratio/{}".format(k)] = epoch_stats["train/{}".format(k)] / epoch_stats[
                    "test/{}".format(k)]
            else:
                epoch_stats["ratio/{}".format(k)] = 0.0

        # At the end of the first epoch, freeze random baseline metrics.
        for k, v in baselines.items():
            for v2 in v["rnd"].values():
                v2.freeze()

        history.append(epoch_stats)
        yield ("epoch", epoch_stats, return_tags)

        if not ok:
            break

    if ok:
        # Add baselines to the history.
        for h in history:
            for split, v in baselines.items():
                for b, v2 in v.items():
                    for m, v3 in v2.items():
                        h["{}/{}_{}".format(split, b, m)] = v3.compute().item()

        for k in metrics["test"].keys():
            if "delta/{}".format(k) in history[-1] and history[-1]["delta/{}".format(k)] > opts["overfitting_threshold"]:
                return_tags.add("Overfitting {} (end)".format(k))

            for s in ["train", "val", "test"]:
                if k in baselines[s]["rnd"] and history[-1]["{}/{}".format(s, k)] <= history[-1]["{}/rnd_{}".format(s, k)] + opts["rnd_threshold"]:
                    return_tags.add("Random guessing {} ({})".format(k, s))
                if k in baselines[s]["mp"] and history[-1]["{}/{}".format(s, k)] <= history[-1]["{}/mp_{}".format(s, k)] + opts["mp_threshold"]:
                    return_tags.add("Most probable guessing {} ({})".format(k, s))

        return_tags.add("Success")

    yield ("end", history, return_tags) # Sentinel signaling the end of training.

def train_step(net, optimizer, batch, metrics, baselines, cls_size, is_pretraining, opts):
    ok = True
    has_been_updated = False
    ret_tags = set()
    imgs, states, var_labels, cons_labels, label, seq_mask, var_mask, cons_mask = batch

    imgs = [i.to(opts["device"]) for i in imgs]
    seq_mask = seq_mask.to(opts["device"])
    states = states.to(opts["device"])

    var_labels = [v.to(opts["device"]) for v in var_labels]  # sorted list: out_z, var_x, var_y, ...
    cons_labels = [p.to(opts["device"]) for p in cons_labels]
    var_mask = [v.to(opts["device"]) for v in var_mask]
    cons_mask = [p.to(opts["device"]) for p in cons_mask]
    label = label.to(opts["device"])

    if opts["use_label_oracle"]:
        true_lbl = var_labels
    else:
        true_lbl = None

    if opts["use_constraint_oracle"]:
        true_cs = cons_labels
    else:
        true_cs = None

    if opts["teacher_forcing"]:
        pred_vars, pred_props, pred_states, pred_label = net.forward_sequence(imgs, seq_mask, true_states=states, true_constraints=true_cs, true_labels=true_lbl)
    else:
        pred_vars, pred_props, pred_states, pred_label = net.forward_sequence(imgs, seq_mask, detach_prev_state=opts["detach_previous_state"], true_constraints=true_cs, true_labels=true_lbl)

    # Rather ugly, but effective, way to flatten timesteps in a way which prevents losses from being updated with averages of averages.
    pv = []
    tv = []
    pp = []
    tp = []
    pv_eval = []
    tv_eval = []
    pp_eval = []
    tp_eval = []

    # Truncate targets if the predicted sequence has been truncated (due to every element in the batch being masked or irrelevant).
    seq_mask = seq_mask[:,:pred_states.size(1) - 1]
    states = states[:,:pred_states.size(1)]
    var_mask = [v[:,:pred_states.size(1) - 1] for v in var_mask]
    cons_mask = [c[:, :pred_states.size(1) - 1] for c in cons_mask]
    var_labels = [v[:, :pred_states.size(1) - 1] for v in var_labels]
    cons_labels = [c[:, :pred_states.size(1) - 1] for c in cons_labels]


    ps = torch.masked_select(pred_states[:, 1:, :], seq_mask.unsqueeze(-1).repeat(1, 1, net.num_states)).reshape(-1, net.num_states)
    ts = torch.masked_select(states[:, 1:], seq_mask)


    for i in range(len(var_labels)):  # out_z, var_x, var_y
        pv_eval.append(torch.masked_select(pred_vars[i], seq_mask.unsqueeze(-1)).reshape(-1, cls_size[i]))
        tv_eval.append(torch.masked_select(var_labels[i], seq_mask))

        if opts["train_on_irrelevant"]:
            pv.append(pv_eval[-1])
            tv.append(tv_eval[-1])
        else:
            pv.append(torch.masked_select(pred_vars[i], var_mask[i].unsqueeze(-1)).reshape(-1, cls_size[i]))
            tv.append(torch.masked_select(var_labels[i], var_mask[i]))

        metrics["train"]["label_acc"].update(i, pv_eval[-1], tv_eval[-1])


    for i in range(len(cons_labels)):
        pp_eval.append(torch.masked_select(pred_props[i], seq_mask))
        tp_eval.append(torch.masked_select(cons_labels[i], seq_mask))
        if opts["train_on_irrelevant"]:
            pp.append(pp_eval[-1])
            tp.append(tp_eval[-1])
        else:
            pp.append(torch.masked_select(pred_props[i], cons_mask[i]))
            tp.append(torch.masked_select(cons_labels[i], cons_mask[i]))

        if torch.sum(cons_labels[i]) >= 0:
            metrics["train"]["const_acc"].update(i, pp_eval[-1], tp_eval[-1])

    # Actual loss computation. Without the previous flattening, this would have required nested loops.
    loss = 0.

    if is_pretraining:
        for i, pred in enumerate(pv):
            loss += (1.0 / len(pv)) * F.cross_entropy(pred, tv[i], reduction="mean")
    else:
        if opts["supervision_lambda"] > 0.:
            for i, pred in enumerate(pv):
                loss += (opts["supervision_lambda"] / len(pv)) * F.cross_entropy(pred, tv[i], reduction="mean")

        if opts["constraint_lambda"] > 0.:
            for i, pred in enumerate(pp):
                loss += (opts["constraint_lambda"] / len(pp)) * F.binary_cross_entropy(pred, tp[i], reduction="mean")

        if opts["successor_lambda"] > 0.:
            loss += opts["successor_lambda"] * F.nll_loss(ps, ts, reduction="mean")

        if opts["sequence_lambda"] > 0.:
            if opts["sequence_loss"] == "bce":
                loss += opts["sequence_lambda"] * F.binary_cross_entropy(pred_label, label)
            else:
                loss += opts["sequence_lambda"] * fuzzy_loss(pred_label, label)


    if torch.isnan(loss):
        ok = False
        ret_tags.add("NaN loss")
    elif torch.isposinf(loss):
        ok = False
        ret_tags.add("Inf loss")
    else:
        optimizer.zero_grad()
        if loss.requires_grad:
            loss.backward()
            has_been_updated = True

    metrics["train"]["loss"].update(loss)
    norms = grad_norm(list(net.parameters()))

    if torch.isnan(norms):
        ok = False
        ret_tags.add("NaN gradient")
    else:
        optimizer.step() # Update weights only if gradient is defined!

    if norms < opts["vanishing_threshold"]:
        #print("Warning: Vanishing gradients. Gradient norm: {}".format(norms))
        ret_tags.add("Vanishing gradient")
    elif norms > opts["exploding_threshold"]:
        #print("Warning: Exploding gradients. Gradient norm: {}".format(norms))
        ret_tags.add("Exploding gradient")

    metrics["train"]["avg_grad_norm"].update(norms)
    metrics["train"]["std_grad_norm"].update(norms)

    if metrics["train"]["std_grad_norm"].compute() > opts["std_threshold"]:
        #print("Warning: High gradient standard deviation.")
        ret_tags.add("High gradient StdDev")

    metrics["train"]["succ_acc"].update(torch.exp(ps), ts)
    metrics["train"]["seq_acc"].update(pred_label, label)

    baselines["train"]["rnd"]["succ_acc"].update(torch.exp(ps), ts)
    baselines["train"]["rnd"]["seq_acc"].update(pred_label, label)
    baselines["train"]["mp"]["succ_acc"].update(torch.exp(ps), ts)
    baselines["train"]["mp"]["seq_acc"].update(pred_label, label)

    return loss, ok, ret_tags, has_been_updated

def eval_step(net, batch, metrics, baselines, split, cls_size, opts):
    imgs, states, var_labels, cons_labels, label, seq_mask, _, _ = batch

    imgs = [i.to(opts["device"]) for i in imgs]
    seq_mask = seq_mask.to(opts["device"])
    states = states.to(opts["device"])
    var_labels = [v.to(opts["device"]) for v in var_labels]
    cons_labels = [p.to(opts["device"]) for p in cons_labels]
    label = label.to(opts["device"])

    if opts["use_label_oracle"]:
        true_lbl = var_labels
    else:
        true_lbl = None

    if opts["use_constraint_oracle"]:
        true_cs = cons_labels
    else:
        true_cs = None

    # Teacher forcing is always disabled in evaluation!
    pred_vars, pred_props, pred_states, pred_label = net.forward_sequence(imgs, seq_mask, true_constraints=true_cs, true_labels=true_lbl)

    # The same trick used to unfold timesteps for the loss is used here for metrics.
    pv = []
    tv = []
    pp = []
    tp = []

    ps = torch.masked_select(pred_states[:, 1:, :], seq_mask.unsqueeze(-1).repeat(1, 1, net.num_states)).reshape(-1, net.num_states)
    ts = torch.masked_select(states[:, 1:], seq_mask)

    for i in range(len(var_labels)):
        pv.append(torch.masked_select(pred_vars[i], seq_mask.unsqueeze(-1)).reshape(-1, cls_size[i]))
        tv.append(torch.masked_select(var_labels[i], seq_mask))
        metrics[split]["label_acc"].update(i, pv[-1], tv[-1])

    for i in range(len(cons_labels)):
        pp.append(torch.masked_select(pred_props[i], seq_mask))
        tp.append(torch.masked_select(cons_labels[i], seq_mask))
        if torch.sum(cons_labels[i]) >= 0:
            metrics[split]["const_acc"].update(i, pp[-1], tp[-1])

    metrics[split]["succ_acc"].update(torch.exp(ps), ts)
    metrics[split]["seq_acc"].update(pred_label, label)

    baselines[split]["rnd"]["succ_acc"].update(torch.exp(ps), ts)
    baselines[split]["rnd"]["seq_acc"].update(pred_label, label)
    baselines[split]["mp"]["succ_acc"].update(torch.exp(ps), ts)
    baselines[split]["mp"]["seq_acc"].update(pred_label, label)