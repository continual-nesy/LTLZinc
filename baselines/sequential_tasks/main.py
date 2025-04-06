import os

import torchvision.transforms
import wandb
import sys
import yaml
import torch
import datetime
import utils

from networks import SequenceClassifier, ScallopE2E
from dataset import LTLZincSequenceDataset
from train import train

# Experiment entrypoint. This script allows to run a single experiment, by passing hyper-parameters as command line
# arguments. It exploits Weights and Biases (W&B) groups and tags to organize experiments in a meaningful way.
# Groups will be given a mnemonic name based on hyper-parameters, in this way, repeating an experiment will automatically
# merge runs into the same group, and aggregated statistics can be computed from the W&B dashboard.
# Events triggered during training (e.g., successful completion, vanishing gradients, etc.) will be associated to a tag,
# to easily filter runs. Tags belong to two categories (informative and errors), a run with errors will be additionally
# flagged by a "Failed" status, instead of the default "Finished", for further filtering.
# This script can be used in three modes:
# - Standalone: do not pass a --wandb_project argument. It will save results locally (if --save True) or print them on screen.
# - W&B single experiment: pass a --wandb_project argument. It will upload results to W&B at the end.
# - W&B sweep experiment: invoke the script from a W&B sweep. In this mode you can fully exploit experiment replicates by
#   sweeping through multiple random seeds (which will be associated to the same group name).


def run(opts, rng):
    """
    Perform an experiment.
    :param opts: Dictionary of hyper-parameters.
    :param rng: Seeded numpy.random.Generator.
    :return: Generator yielding tuples (dict of epoch statistics, set of epoch tags).
    """

    annotations_path = "{}/{}".format(opts["prefix_path"], opts["annotations_path"])
    task_dir = "{}/{}/{}".format(annotations_path, opts["task"], opts["task_seed"])

    task_opts = {}
    with open("{}/automaton.yml".format(task_dir), "r") as file:
        task_opts["automaton"] = yaml.safe_load(file)

    with open("{}/domain_values.yml".format(task_dir), "r") as file:
        task_opts["domain_values"] = yaml.safe_load(file)

    with open("{}/mappings.yml".format(task_dir), "r") as file:
        task_opts["mappings"] = yaml.safe_load(file)

    classes = utils.domains_to_class_ids(task_opts)

    if opts["standardize_inputs"]:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ConvertImageDtype(torch.float),
            torchvision.transforms.Normalize(mean=0.0, std=1.0, inplace=True)
        ])
    else:
        transform = None

    train_ds = LTLZincSequenceDataset(opts["prefix_path"], task_dir, classes, "train", opts["use_random_samples"], rng, transform=transform)
    val_ds = LTLZincSequenceDataset(opts["prefix_path"], task_dir, classes, "val", False, rng, transform=transform)
    test_ds = LTLZincSequenceDataset(opts["prefix_path"], task_dir, classes, "test", False, rng, transform=transform)

    if opts["scallop_e2e"]:
        model = ScallopE2E(opts, task_opts, classes)
    else:
        model = SequenceClassifier(opts, task_opts, classes, opts["oracle_noise"], opts["oracle_type"])

    for (status, history, return_tags) in train(model, classes, train_ds, val_ds, test_ds, opts):
        # Postpone end-of-training signaling, so that the process is not killed before saving results.
        if status != "end":
            yield (status, history, return_tags)

    if opts["save"]:
        op = "{}/{}".format(opts["prefix_path"], opts["output_path"])
        if not os.path.exists(op):
            os.mkdir(op)

        # Save model weights.
        filename = "{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), experiment_name)
        torch.save(model.state_dict(), "{}/{}.pt".format(op, filename))

        # Save hyper-parameters, to reconstruct the correct model.
        with open("{}/{}.yml".format(op, filename), "w") as file:
            yaml.safe_dump(opts, file)

        # Save results.
        with open("{}/{}_results.yml".format(op, filename), "w") as file:
            yaml.safe_dump({"history": history, "tags": return_tags}, file)

    yield ("end", history, return_tags)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Warning: no arguments were provided. Will run with default values.")

    arg_parser = utils.get_arg_parser()
    opts = vars(arg_parser.parse_args())

    utils.preflight_checks(opts)
    experiment_name = utils.generate_name(opts, "cows")
    run_experiment = utils.prune_hyperparameters(opts, arg_parser) or not opts["abort_irrelevant"]


    # Override experiment device if environment variable is set.
    if os.environ.get('DEVICE') is not None:
        opts['device'] = os.environ.get('DEVICE')


    wb = None # Don't use W&B.
    if "WANDB_SWEEP_ID" in os.environ: # Use W&B in sweep mode.
        wb = wandb.init(group=experiment_name, config=opts)
    elif opts['wandb_project'] is not None: # Use W&B in standalone mode.
        wb = wandb.init(project=opts['wandb_project'], group=experiment_name, config=opts)


    if run_experiment:
        rng = utils.set_seed(opts["seed"])

        with utils.Watchdog(run, opts["heartbeat"], opts, rng) as wd:
            history, return_tags = wd.listen(opts["epoch_timeout"] * 60)

        if wb is not None:
            wb.tags = sorted(return_tags)
            for i, h in enumerate(history):
                wb.log(data=h, step=i)

            if "Success" not in return_tags:
                wb.finish(exit_code=1)
            else:
                wb.finish()

        else: # If W&B is not enabled, at least print results on screen.
            print({"history": history, "tags": return_tags})

    else:
        if wb is not None:
            wb.tags = ["Irrelevant hyperparameters"]
            wb.finish(exit_code=1)