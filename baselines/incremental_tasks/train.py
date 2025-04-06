import torch
from torch.utils.data import DataLoader
import torcheval.metrics
import tqdm
import time
import torch.nn.functional as F
import copy

from metrics import StdDev, grad_norm, avg_accuracy, avg_forgetting, forward_transfer, backward_transfer
import quadprog
import numpy as np


def train(net, buffers, teachers, train_ds, val_ds, test_ds, target_metrics, opts, task_opts):
    """
    Train the model on the provided datasets. If some events are triggered during training, it collects them on a set of flags.
    Generator returning tuples in the form (phase, metrics, tags) for each phase. Phase is a string in ["start", "epoch", "task", "end"]
    In order to correctly synchronize with the watchdog, it must yield two sentinels:
    ("start", None, None) signals the end of the initialization phase and the start of proper training.
    ("end", full_history, tags) signals the end of training, possibly providing additional tags (e.g., "Success" or "NaN loss").
    :param net: torch.nn.Module to train.
    :param buffers: Dictionary of replay buffers (one for each knowledge bin), or None.
    :param teachers: Dictionary of teachers for teacher-student distillation (one for each knowledge bin), or None.
    :param train_ds: Train dataset.
    :param val_ds: Validation dataset.
    :param test_ds: Test dataset.
    :param target_metrics: List of classes of interest for focused metrics.
    :param opts: Dictionary of hyper-parameters.
    :param task_opts: Dictionary of task-specific parameters.
    :return: Generator yielding tuples (phase, metrics evaluated for the current epoch, set of events triggered during the current epoch).
    """

    net.to(opts["device"])
    target_metrics = torch.tensor(target_metrics, device=opts["device"])

    train_dls = [DataLoader(ds, batch_size=opts["batch_size"], shuffle=opts["shuffle"]) for ds in train_ds]

    val_dls = [DataLoader(ds, batch_size=opts["eval_batch_size"], shuffle=False) for ds in val_ds]
    test_dls = [DataLoader(ds, batch_size=opts["eval_batch_size"], shuffle=False) for ds in test_ds]

    eval_sets = {"val": val_dls, "test": test_dls}
    if opts["eval_train"]:
        eval_sets["train"] = [DataLoader(ds, batch_size=opts["eval_batch_size"], shuffle=opts["shuffle"]) for ds in train_ds]

    if opts["distillation_source"] != "none" and opts["knowledge_availability"] == "none":
        teacher = copy.deepcopy(net)
        teacher.to(opts["device"])
    else:
        teacher = None

    if teachers is not None:
        for k, v in teachers.items():
            if opts["lr"] > 0.0:
                teachers[k] = {"net": v.to(opts["device"]), "optimizer": torch.optim.SGD(net.parameters(), lr=opts["lr"]), "trained": False}
            else:
                teachers[k] = {"net": v.to(opts["device"]), "optimizer": torch.optim.Adam(net.parameters(), lr=-opts["lr"]), "trained": False}

    if opts["knowledge_availability"] == "none":
        full_knowledge = set()
    elif opts["knowledge_availability"] == "states":
        full_knowledge = set(["s_{}".format(s) for s in task_opts["automaton"]["states"]])
    else:
        full_knowledge = set(task_opts["mappings"].keys())

    if opts["buffer_loss"] == "gem":
        gem_dict = {"grad_dims": []}

        gem_dict["grad_dims"] = []
        for param in net.parameters():
            gem_dict["grad_dims"].append(param.data.numel())
        gem_dict["grads"] = {k: torch.empty(sum(gem_dict["grad_dims"]), device=opts["device"]) for k in full_knowledge}
        gem_dict["cur_grads"] = torch.empty(sum(gem_dict["grad_dims"]), device=opts["device"])
    else:
        gem_dict = None

    assert len(train_ds) == len(val_ds) == len(test_ds), \
        "Training validation and test sets must contain the same number of tasks, found: {}, {}, {}".format(len(train_ds), len(val_ds), len(test_ds))

    if opts["lr"] > 0.0:
        optimizer = torch.optim.SGD(net.parameters(), lr=opts["lr"])
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=-opts["lr"])

    if opts["grad_clipping"] > 0.0:
        for p in net.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -opts["grad_clipping"], opts["grad_clipping"]))

    history = []
    ok = True
    return_tags = set()

    metrics = {}
    metrics["train"] = {}

    # Additional metrics.
    metrics["train"]["loss"] = torcheval.metrics.Mean(device=opts["device"])
    metrics["train"]["avg_grad_norm"] = torcheval.metrics.Mean(device=opts["device"])
    metrics["train"]["std_grad_norm"] = StdDev(device=opts["device"])

    metrics["val"] = {}
    metrics["test"] = {}

    for split in eval_sets.keys():
        metrics[split]["acc_matrix"] = torch.zeros((len(train_ds), len(train_ds)), dtype=float, device=opts["device"])
        metrics[split]["focused_acc_matrix"] = torch.zeros((len(train_ds), len(train_ds)), dtype=float, device=opts["device"])

    buffer_batch = None

    yield ("start", None, None)  # Sentinel to start the watchdog heartbeat.

    for t in (bar := tqdm.trange(len(train_ds), position=0, desc="task", disable=opts["verbose"] < 1)):
        if opts["knowledge_availability"] == "none":
            knowledge_str = "No knowledge"
            trainable_knowledge = set()
        elif opts["knowledge_availability"] == "states":
            knowledge_str = "Cur. state: {}".format(train_dls[t].dataset.curr_state)
            trainable_knowledge = {"s_{}".format(train_dls[t].dataset.curr_state)}
        else:
            knowledge_str = "Cur. preds: {}/{}".format(len(train_dls[t].dataset.curr_preds),
                                                       len(train_dls[t].dataset.constraints))
            trainable_knowledge = set(train_dls[t].dataset.curr_preds)

        if opts["modular_net"]:
            net.train_layers(trainable_knowledge)

        if teacher is not None: # Self-distillation.
            teacher.load_state_dict(net.state_dict())
            teacher.eval()

        start_time = time.time()
        for e in tqdm.trange(opts["epochs"], position=1, desc="epoch", leave=False, disable=opts["verbose"] < 2):
            for m in ["loss", "avg_grad_norm", "std_grad_norm"]:
                metrics["train"][m].reset()

            net.train()
            if teachers is not None:
                for k, v in teachers.items():
                    v["net"].train(k in trainable_knowledge)

            for batch in (tqdm.tqdm(train_dls[t], position=2, desc="batch", leave=False, ncols=0, disable=opts["verbose"] < 3)):
                if buffers is not None:
                    for k, v in buffers.items():
                        if k in trainable_knowledge:
                            v.observe(batch)


                if t > 0:
                    if opts["buffer_loss"] != "none" or opts["distillation_source"] in ["buffer", "both"]:
                        buffer_batch = {}
                        if buffers is not None:
                            for k, v in buffers.items():
                                if k not in trainable_knowledge:
                                    if opts["knowledge_availability"] == "none":
                                        tmp = v.get_random_batch(opts["buffer_batch_size"])
                                    else:
                                        tmp = v.get_random_batch(
                                            opts["buffer_batch_size"] // (len(full_knowledge) - len(trainable_knowledge)))

                                    if len(tmp[0]) > 0:
                                        buffer_batch[k] = [torch.stack(tmp[0], dim=0), torch.stack(tmp[1], dim=0), torch.stack(tmp[2], dim=0)]

                        # In case no sample has been buffered, skip replay for this epoch.
                        if len(buffer_batch.keys()) == 0:
                            buffer_batch = None


                loss, ok, step_tags = train_step(net, optimizer, batch, buffer_batch, gem_dict, teachers, teacher, metrics, trainable_knowledge, t == 0, opts)

                return_tags.update(step_tags)

                # If there was an error, abort training.
                if not ok:
                    break

            yield ("epoch", None, return_tags)

            bar.set_postfix_str(
                "Loss: {:.02f}, avg_Loss: {:.02f}, avg_dLoss: {:.02f}, std_dLoss: {:.02f}, [{}]".format(
                    loss,
                    metrics["train"]["loss"].compute(), metrics["train"]["avg_grad_norm"].compute(),
                    metrics["train"]["std_grad_norm"].compute(), knowledge_str
                ))

        times = {"train": time.time() - start_time}

        with torch.no_grad():
            net.eval()

            for split, dls in eval_sets.items():
                start_time = time.time()
                for t2 in tqdm.trange(len(dls), position=2, desc="eval task", leave=False, ncols=0,
                                      disable=opts["verbose"] < 3):
                    metrics[split]["acc_matrix"][t, t2] = 0.0
                    metrics[split]["focused_acc_matrix"][t, t2] = 0.0
                    focused_samples = 0
                    for batch in (tqdm.tqdm(dls[t2], position=3, desc="eval batch", leave=False, ncols=0,
                                            disable=opts["verbose"] < 4)):
                        num_focused_samples = eval_step(net, batch, metrics[split]["acc_matrix"],
                                                        metrics[split]["focused_acc_matrix"], target_metrics, t, t2,
                                                        opts)
                        focused_samples += num_focused_samples

                    metrics[split]["acc_matrix"][t, t2] /= len(dls[t2].dataset) * len(net.heads.keys())

                    if focused_samples > 0:
                        metrics[split]["focused_acc_matrix"][t, t2] /= focused_samples
                    else:
                        metrics[split]["focused_acc_matrix"][t, t2] = float('nan')

                # Avoid overriding training times.
                if split != "train":
                    times[split] = time.time() - start_time

        task_stats = {}
        for split in ["train", "val", "test"]:
            task_stats["{}/time".format(split)] = times[split]

            if split in eval_sets.keys():
                task_stats["{}/avg_accuracy".format(split)] = avg_accuracy(metrics[split]["acc_matrix"][:t+1,:])
                task_stats["{}/avg_forgetting".format(split)] = avg_forgetting(metrics[split]["acc_matrix"][:t+1,:])
                task_stats["{}/fw_transfer".format(split)] = forward_transfer(metrics[split]["acc_matrix"])
                task_stats["{}/bw_transfer".format(split)] = backward_transfer(metrics[split]["acc_matrix"])

                task_stats["{}/focused_avg_accuracy".format(split)] = avg_accuracy(
                    metrics[split]["focused_acc_matrix"][:t+1,:])
                task_stats["{}/focused_avg_forgetting".format(split)] = avg_forgetting(
                    metrics[split]["focused_acc_matrix"][:t+1,:])

            for k, v in metrics[split].items():
                if torch.is_tensor(v):
                    task_stats["{}/{}".format(split, k)] = v.tolist()
                elif isinstance(v, torcheval.metrics.Metric):
                    task_stats["{}/{}".format(split, k)] = v.compute().item()
                else:
                    task_stats["{}/{}".format(split, k)] = v

        history.append(task_stats)
        yield ("task", task_stats, return_tags)

        if not ok:
            break

    if ok:
        return_tags.add("Success")

    yield ("end", history, return_tags)  # Sentinel signaling the end of training.


def compute_cls_loss(net, imgs, var_labels, var_mask, opts):
    logits = net(imgs)

    if not opts["train_on_irrelevant"]:
        var_labels = torch.masked_select(var_labels, var_mask).reshape(var_labels.shape)
        logits = torch.masked_select(logits, var_mask.unsqueeze(1).repeat(1, logits.size(1), 1)).reshape(
            logits.shape)

    loss = F.cross_entropy(logits, var_labels.squeeze(), reduction="mean")

    return loss

def compute_kld_loss(student, teacher, imgs):
    teacher_logits = teacher(imgs).detach()
    student_logits = student(imgs)

    # Note: While KL divergence is normally KLD(pred, true), in knowledge distillation it is KLD(specialized_model, generalized_model).
    loss = F.relu(F.kl_div(teacher_logits, student_logits, log_target=True, reduction="batchmean"))

    return loss

def store_grad(params,  gem_dict, slot):
    if slot is not None:
        gem_dict["grads"][slot].fill_(0.0)
        for i, param in enumerate(params()):
            if param.grad is not None:
                beg = 0 if i == 0 else sum(gem_dict["grad_dims"][:i])
                en = sum(gem_dict["grad_dims"][:i + 1])
                gem_dict["grads"][slot][beg: en].copy_(param.grad.data.view(-1))
    else:
        gem_dict["cur_grads"].fill_(0.0)
        for i, param in enumerate(params()):
            if param.grad is not None:
                beg = 0 if i == 0 else sum(gem_dict["grad_dims"][:i])
                en = sum(gem_dict["grad_dims"][:i + 1])
                gem_dict["cur_grads"][beg: en].copy_(param.grad.data.view(-1))


def overwrite_grad(params, newgrad, grad_dims):
    for i, param in enumerate(params()):
        if param.grad is not None:
            beg = 0 if i == 0 else sum(grad_dims[:i])
            en = sum(grad_dims[:i + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    n_rows = memories_np.shape[0]
    self_prod = np.dot(memories_np, memories_np.transpose())
    self_prod = 0.5 * (self_prod + self_prod.transpose()) + np.eye(n_rows) * eps
    grad_prod = np.dot(memories_np, gradient_np) * -1
    G = np.eye(n_rows)
    h = np.zeros(n_rows) + margin
    v = quadprog.solve_qp(self_prod, grad_prod, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.from_numpy(x).view(-1, 1))

def train_step(net, optimizer, batch, buffer_batch, gem_dict, teachers, old_net, metrics, trainable_knowledge, first_epoch, opts):
    ok = True
    ret_tags = set()

    imgs, var_labels, var_mask = batch

    imgs = imgs.to(opts["device"])

    var_labels = var_labels.to(opts["device"])
    var_mask = var_mask.to(opts["device"])

    if buffer_batch is not None and not first_epoch:
        buffer_batch = {k: [x.to(opts["device"]) for x in v] for k, v in buffer_batch.items()}

        if opts["buffer_loss"] == "gem":
            for k, v in buffer_batch.items():
                optimizer.zero_grad()
                gem_loss = compute_cls_loss(net, v[0], v[1], v[2], opts)

                gem_loss.backward()
                store_grad(net.parameters, gem_dict, k)

    loss = 0.
    if opts["supervision_lambda"] > 0.:
        loss += opts["supervision_lambda"] * compute_cls_loss(net, imgs, var_labels, var_mask, opts)

    if opts["distil_lambda"] > 0. and not first_epoch:
        kld_loss = 0.
        if opts["distillation_source"] == "batch" or opts["distillation_source"] == "both":
            if old_net is not None:
                kld_loss += compute_kld_loss(net, old_net, imgs)
            elif teachers is not None:
                for k, v in teachers.items():
                    if k in trainable_knowledge:
                        kld_loss += compute_kld_loss(net, v["net"], imgs) / len(trainable_knowledge)

        if (opts["distillation_source"] == "buffer" or opts["distillation_source"] == "both") and buffer_batch is not None and not first_epoch:
            if old_net is not None:
                for bv in buffer_batch.values():
                    kld_loss += compute_kld_loss(net, old_net, bv[0])
            elif teachers is not None:
                scale = 1.0 / sum([1 for v in teachers.values() if v["trained"]])
                for k, tv in teachers.items():
                    if k not in trainable_knowledge and tv["trained"]:
                        kld_loss += compute_kld_loss(net, tv["net"], buffer_batch[k][0]) * scale

        if opts["distillation_source"] == "both" and buffer_batch is not None and not first_epoch:
            kld_loss /= 2.0

        loss += opts["distil_lambda"] * kld_loss

    if opts["replay_lambda"] > 0. and opts["buffer_loss"] == "ce" and buffer_batch is not None and not first_epoch:
        if opts["buffer_loss"] == "ce":
            for v in buffer_batch.values():
                loss += opts["replay_lambda"] * compute_cls_loss(net, v[0], v[1], v[2], opts) / len(buffer_batch.keys())

    if torch.isnan(loss):
        ok = False
        ret_tags.add("NaN loss")
    elif torch.isposinf(loss):
        ok = False
        ret_tags.add("Inf loss")
    else:
        optimizer.zero_grad()
        loss.backward()
        if opts["buffer_loss"] == "gem" and buffer_batch is not None and not first_epoch:
            store_grad(net.parameters, gem_dict, None)
            old_grads = torch.stack([gem_dict["grads"][k] for k in sorted(gem_dict["grads"].keys())]).T
            dotp = torch.mm(gem_dict["cur_grads"].unsqueeze(0), old_grads)

            if (dotp < 0).sum() != 0:
                margin = opts["replay_lambda"] / (opts["supervision_lambda"] + opts["distil_lambda"] + opts["replay_lambda"])
                project2cone2(gem_dict["cur_grads"].unsqueeze(1), old_grads, margin=margin, eps=opts["eps"])
                overwrite_grad(net.parameters, gem_dict["cur_grads"], gem_dict["grad_dims"])


    metrics["train"]["loss"].update(loss)
    norms = grad_norm(list(net.parameters()))


    if torch.isnan(norms):
        ok = False
        ret_tags.add("NaN gradient")
    else:
        optimizer.step() # Update weights only if gradient is defined!

    if teachers is not None:
        for k, v in teachers.items():
            if k in trainable_knowledge: # Train current teachers.
                teacher_loss = compute_cls_loss(v["net"], imgs, var_labels, var_mask, opts)

                v["optimizer"].zero_grad()
                teacher_loss.backward()
                v["optimizer"].step()
                v["trained"] = True


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

    return loss, ok, ret_tags

def eval_step(net, batch, acc_matrix, focused_acc_matrix, target_metrics, t, t2, opts):
    imgs, var_labels, var_mask = batch

    imgs = imgs.to(opts["device"])
    logits = net(imgs)

    var_labels = var_labels.to(opts["device"]).squeeze()
    var_mask = var_mask.to(opts["device"]).squeeze()

    preds = torch.argmax(logits, dim=1)


    if opts["train_on_irrelevant"]:
        preds = torch.masked_select(preds, var_mask)
        var_labels = torch.masked_select(var_labels, var_mask)

    acc_matrix[t, t2] += torch.count_nonzero(preds == var_labels)

    focused_labels = torch.masked_select(var_labels, torch.isin(var_labels, target_metrics))
    focused_preds = torch.masked_select(preds, torch.isin(var_labels, target_metrics))
    if len(focused_labels) > 0:
        focused_acc_matrix[t, t2] += torch.count_nonzero(focused_preds == focused_labels)
    focused_divisor = len(focused_preds)

    return focused_divisor
