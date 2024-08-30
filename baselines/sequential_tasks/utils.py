import random
import time
import numpy as np
import torch
import hashlib
import argparse
import os
import re

import multiprocessing
import queue


from mnemonics import __mnemonic_names__

class ArgNumber:
    """
    Simple numeric argument validator.
    """
    def __init__(self, number_type: type(int) | type(float),
                 min_val: int | float | None = None,
                 max_val: int | float | None = None):
        self.__number_type = number_type
        self.__min = min_val
        self.__max = max_val
        if number_type not in [int, float]:
            raise argparse.ArgumentTypeError("Invalid number type (it must be int or float)")
        if not ((self.__min is None and self.__max is None) or
                (self.__max is None) or (self.__min is None) or (self.__min is not None and self.__min < self.__max)):
            raise argparse.ArgumentTypeError("Invalid range")

    def __call__(self, value: int | float | str) -> int | float:
        try:
            val = self.__number_type(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid value specified, conversion issues! Provided: {value}")
        if self.__min is not None and val < self.__min:
            raise argparse.ArgumentTypeError(f"Invalid value specified, it must be >= {self.__min}")
        if self.__max is not None and val > self.__max:
            raise argparse.ArgumentTypeError(f"Invalid value specified, it must be <= {self.__max}")
        return val


class ArgBoolean:
    """
    Simple Boolean argument validator.
    """
    def __call__(self, value: str | bool | int) -> bool:
        if isinstance(value, str):
            val = value.lower().strip()
            if val != "true" and val != "false" and val != "yes" and val != "no":
                raise argparse.ArgumentTypeError(f"Invalid value specified: {value}")
            val = True if (val == "true" or val == "yes") else False
        elif isinstance(value, int):
            if value != 0 and value != 1:
                raise argparse.ArgumentTypeError(f"Invalid value specified: {value}")
            val = value == 1
        elif isinstance(value, bool):
            val = value
        else:
            raise argparse.ArgumentTypeError(f"Invalid value specified (expected boolean): {value}")
        return val

def get_arg_parser():
    """
    Build the argument parser. Define here all the experiment hyper-parameters and house-keeping arguments
    (e.g., whether to save weights at the end of training). For readability, we usually group them by category.
    To reduce the (already large) number of arguments, we try to group related HPs with a colon-separated string
    (e.g., "mlp:NUM_LAYERS:ACTIVATION") and we split it during parsing. This is also beneficial to reduce grid searches.

    generate_readme_stub.py will parse the help field automatically. Default values should be put at the end of the string,
    as "(default: value)", which will be deleted, or as "(comment; default: value)", which will become "(comment)".
    :return: An argparse.ArgumentParser
    """
    arg_parser = argparse.ArgumentParser()
    # Dataset parameters.
    arg_parser.add_argument('--prefix_path', help="Path to the root of the project (default: '../..')", type=str,
                            default="../..")
    arg_parser.add_argument('--annotations_path',
                            help="Path to the annotation folder, relative to root (default: 'benchmark/tasks')",
                            type=str, default="benchmark/tasks")
    arg_parser.add_argument('--output_path',
                            help="Path to the output folder, relative to root (default: 'outputs')",
                            type=str, default="outputs")
    arg_parser.add_argument('--use_random_samples',
                            help="Ignore annotated image paths and sample new ones with the same label (default: False)",
                            type=ArgBoolean(), default=False)
    arg_parser.add_argument('--task', help="Name of the task to solve (default: 'task1')", type=str, default="task1")

    # Training parameters.
    arg_parser.add_argument("--pretraining_lr", help="Learning rate for the pre-training phase (positive: SGD, negative: Adam; default: -1e-3)",
                            type=ArgNumber(float), default=-1e-3)
    arg_parser.add_argument("--lr", help="Learning rate (positive: SGD, negative: Adam; default: -1e-4)",
                            type=ArgNumber(float), default=-1e-4)
    arg_parser.add_argument("--epochs", help="Training epochs (default: 10)", type=ArgNumber(int, min_val=1),
                            default=10)
    arg_parser.add_argument("--pretraining_epochs",
                            help="Pre-training epochs, these will apply only supervision loss with a fixed lambda=1.0 (default: 0)",
                            type=ArgNumber(int, min_val=0), default=0)
    arg_parser.add_argument("--batch_size", help="Batch size (default: 32)", type=ArgNumber(int, min_val=1), default=32)
    arg_parser.add_argument("--supervision_lambda", help="Weight for direct supervision (default: 0.0)",
                            type=ArgNumber(float, min_val=0.0), default=0.0)
    arg_parser.add_argument("--constraint_lambda", help="Weight for constraint supervision (default: 0.0)",
                            type=ArgNumber(float, min_val=0.0), default=0.0)
    arg_parser.add_argument("--successor_lambda", help="Weight for next-state supervision (default: 0.0)",
                            type=ArgNumber(float, min_val=0.0), default=0.0)
    arg_parser.add_argument("--sequence_lambda", help="Weight for end-to-end sequence supervision (default: 0.0)",
                            type=ArgNumber(float, min_val=0.0), default=1.0)
    arg_parser.add_argument('--sequence_loss',
                            help="Type of loss function for sequence classification in {'bce', 'fuzzy'} (default: 'bce')",
                            type=str, default="bce", choices=["bce", "fuzzy"])
    arg_parser.add_argument('--teacher_forcing',
                            help="Force ground truth next state during training (False) or let the network evolve on its own (True) (default: False)",
                            type=ArgBoolean(), default=False)
    arg_parser.add_argument('--detach_previous_state',
                            help="During backpropagation detach the computation graph at each time-step (default: False)",
                            type=ArgBoolean(), default=False)
    arg_parser.add_argument('--train_on_irrelevant',
                            help="Apply the loss also on irrelevant annotations (default: False)", type=ArgBoolean(),
                            default=False)
    arg_parser.add_argument('--grad_clipping', help="Norm for gradient clipping, disable if 0.0 (default: 0.0)",
                            type=ArgNumber(float, min_val=0.0), default=0.0)


    # Model parameters.
    arg_parser.add_argument('--backbone_module',
                            help="Perceptual backbone in {'independent', 'dataset'} (default: 'dataset')",
                            type=str, default="dataset", choices=["independent", "dataset"])
    arg_parser.add_argument('--constraint_module',
                            help="Constraint module in {'mlp:NUM_HIDDEN_NEURONS', 'scallop:PROVENANCE:TRAIN_K:TEST_K'} (default: 'mlp:8')",
                            type=str, default="mlp:8")
    arg_parser.add_argument('--dfa_module',
                            help="Automaton module in {'mlp:HIDDEN_NEURONS', 'gru:HIDDEN_NEURONS', 'scallop:PROVENANCE:TRAIN_K:TEST_K', 'fuzzy:SEMIRING', 'sddnnf:SEMIRING'} (default: 'mlp:8')",
                            type=str, default="mlp:8")
    arg_parser.add_argument('--scallop_e2e',
                            help="Use an End-to-End Scallop program for constraints and automaton, instead of two disjoint programs (default: False)",
                            type=ArgBoolean(), default=False)
    arg_parser.add_argument('--use_label_oracle',
                            help="Replace the backbone module with ground truth annotations (default: False)", type=ArgBoolean(),
                            default=False)
    arg_parser.add_argument('--use_constraint_oracle',
                            help="Replace the constraint module with ground truth annotations (default: False)", type=ArgBoolean(),
                            default=False)
    arg_parser.add_argument('--calibrate',
                            help="Use additional temperature parameters to calibrate probabilities (default: False)",
                            type=ArgBoolean(), default=False)

    # Experiment parameters.
    arg_parser.add_argument('--seed',
                            help="Integer seed for random generator (if negative, use timestamp; default: -1)",
                            type=int, default=-1)
    arg_parser.add_argument('--epoch_timeout',
                            help="Timeout for each epoch, in minutes (disable if 0; default: 0)",
                            type=ArgNumber(int, min_val=0), default=0)
    arg_parser.add_argument('--device',
                            help="Device to use, no effect if environment variable DEVICE is set (default: 'cpu')",
                            type=str, default="cpu")
    arg_parser.add_argument('--save', help="Save network weights and results locally (default: False)",
                            type=ArgBoolean(), default=False)
    arg_parser.add_argument('--abort_irrelevant',
                            help="Abort irrelevant combinations of hyper-parameters (useful when sweeping grid searches; default: True)",
                            type=ArgBoolean(), default=True)
    arg_parser.add_argument('--verbose',
                            help="Amount of output (0: no output, 1: epoch summary, 2: full; default: 1)",
                            type=int, choices=[0, 1, 2], default=1)
    arg_parser.add_argument("--wandb_project", help="Use W&B, this is the project name (default: None)", type=str,
                            default=None)

    # Magic numbers.
    arg_parser.add_argument('--eps',
                            help="Epsilon for numerical stability (default: 1e-7)",
                            type=ArgNumber(float, min_val=0), default=1e-7)
    arg_parser.add_argument('--neginf',
                            help="Negative infinity for numerical stability (default: -1e10)",
                            type=ArgNumber(float, max_val=0), default=-1e10)
    arg_parser.add_argument('--overfitting_threshold',
                            help="Threshold triggering an overfitting tag (default: 0.5)",
                            type=ArgNumber(float, min_val=0), default=0.5)
    arg_parser.add_argument('--vanishing_threshold',
                            help="Threshold triggering a vanishing gradient tag (default: 1e-5)",
                            type=ArgNumber(float, min_val=0), default=1e-5)
    arg_parser.add_argument('--exploding_threshold',
                            help="Threshold triggering an exploding gradient tag (default: 1e7)",
                            type=ArgNumber(float, min_val=0), default=1e7)
    arg_parser.add_argument('--std_threshold',
                            help="Threshold triggering an high gradient standard deviation tag (default: 200.0)",
                            type=ArgNumber(float, min_val=0), default=200.0)
    arg_parser.add_argument('--rnd_threshold',
                            help="Threshold triggering a random guessing tag (default: 0.1)",
                            type=ArgNumber(float, min_val=0), default=0.1)
    arg_parser.add_argument('--mp_threshold',
                            help="Threshold triggering a most probable guessing tag (default: 0.1)",
                            type=ArgNumber(float, min_val=0), default=0.1)
    arg_parser.add_argument('--heartbeat',
                            help="Heartbeat for the watchdog timer in seconds (default: 10)",
                            type=ArgNumber(int, min_val=1), default=10)

    return arg_parser

def get_unhashable_opts():
    """
    Explicit list of arguments which should NOT be used for the unique identifier computation.
    :return: The list of excluded arguments.
    """
    return ["wandb_project", "save", "seed", "device", "annotations_path", "data_path", "eps", "neginf",
            "overfitting_threshold", "vanishing_threshold", "exploding_threshold", "std_threshold", "rnd_threshold",
            "mp_threshold", "heartbeat", "epoch_timeout", "verbose"]

def generate_name(opts, mnemonics=None):
    """
    Get a deterministic identifier for a group of experiments, based on the MD5 hash of relevant hyper-parameters.
    :param opts: Dictionary of hyper-parameters. The identifier will be computed based on its content.
    :param mnemonics: Which namespace to use.
    :return: A unique string identifier for the current experiment.
    """

    groupable_opts = set(opts.keys()).difference(get_unhashable_opts())

    if mnemonics is None:
        if "wandb_project" not in opts or opts["wandb_project"] is None:
            mnemonics = "cows"
        else:
            project_hash = int(hashlib.md5(opts["wandb_project"].encode("utf-8")).hexdigest(), 16)
            mnemonics = sorted(__mnemonic_names__.keys())[project_hash % len(__mnemonic_names__)]

    unique_id = ""
    for o in sorted(groupable_opts):
        if isinstance(opts[o], float):
            unique_id += "{:.6f}".format(opts[o])
        else:
            unique_id += str(opts[o])
    hash = hashlib.md5(unique_id.encode("utf-8")).hexdigest()

    assert isinstance(mnemonics, str) or isinstance(mnemonics, list), \
        "Mnemonics must be a string or a list of strings. Found {}.".format(type(mnemonics))
    if isinstance(mnemonics, str):
        assert mnemonics in __mnemonic_names__.keys(), \
            "Unknown mnemonics group. It must be one of {{{}}}".format(", ".join(__mnemonic_names__.keys()))
        # Salting is required to minimize the risk of collisions on small lists of mnemonic names.
        salted_hash = hashlib.md5("{}{}".format(unique_id, mnemonics).encode("utf-8")).hexdigest()
        idx = int(hash, 16) % len(__mnemonic_names__[mnemonics])
        out = "{}_{}".format(__mnemonic_names__[mnemonics][idx], salted_hash[:len(salted_hash) // 4])
    else:
        salted_hash = hashlib.md5("{}{}".format(unique_id, mnemonics[0]).encode("utf-8")).hexdigest()
        idx = int(hash, 16) % len(mnemonics)
        out = "{}_{}".format(mnemonics[idx], salted_hash[:len(salted_hash) // 4])

    return out


def set_seed(seed):
    """
    Set the seed of random number generators in a paranoid manner.
    Note that some code may still be non-deterministic, especially in imported libraries.
    :param seed: Random seed, if < 0, it will use the current timestamp.
    :return: A seeded numpy.random.Generator.
    """
    seed = int(time.time()) if seed < 0 else int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return np.random.default_rng(seed)

def preflight_checks(opts):
    """
    Perform important checks before starting an experiment. This function should fail for experiment-breaking errors
    (e.g., missing dataset folders, invalid values of hyper-parameters, etc.).
    Optionally modifies the hyper-parameters dictionary to convert some options from human-friendly to machine-friendly.
    :param opts: Dictionary of hyper-parameters.
    :return: The modified dictionary of HPs.
    """

    # Hyper-parameter checks. Note: most checks are automatically performed by argparse.
    valid_provenances = ['difftopkproofs', 'diffminmaxprob', 'diffaddmultprob', 'diffnandmultprob', 'diffmaxmultprob',
                         'diffnandminprob', 'diffsamplekproofs', 'difftopbottomkclauses']

    cm = opts["constraint_module"].split(":")
    assert cm[0] in ["mlp", "scallop"], \
        "Invalid constraint module, it must be either of {{'mlp', 'scallop'}}, found {}.".format(cm[0])
    if cm[0] == "mlp":
        assert len(cm) == 2, \
            "The MLP constraint module takes exactly 1 hyper-parameter (number of neurons in the hidden layer), found {}.".format(cm[1:])
        assert int(cm[1], 10) >= 1, \
            "The number of hidden neurons for the constraint module must be an integer >= 1, found {}.".format(cm[1])
        opts["constraint_module"] = {"type": "mlp", "neurons": int(cm[1], 10)}
    elif cm[0] == "scallop":
        assert len(cm) == 4, \
            "The Scallop constraint module takes exactly 3 hyper-parameters (provenance semiring, " + \
            "top-k value for training, top-k value for inference), found {}.".format(cm[1:])

        assert cm[1] in valid_provenances, \
            "The supported provenance semirings are {{{}}}, found {}.".format(", ".join(["'{}'".format(v) for v in valid_provenances]), cm[1])
        assert int(cm[2], 10) >= 1 and int(cm[3], 10) >= 1, \
            "Top-k values for Scallop must be positive integers, found train_k={}, test_k={}".format(cm[2], cm[3])
        opts["constraint_module"] = {"type": "scallop", "provenance": cm[1], "train_k": int(cm[2], 10), "test_k": int(cm[3], 10)}

    dm = opts["dfa_module"].split(":")
    assert dm[0] in ["mlp", "gru", "scallop", "fuzzy", "sddnnf"], \
        "Invalid automaton module, it must be one of {{'mlp:N', 'gru:N', 'scallop:PROVENANCE:TRAIN_K:TEST_K', 'fuzzy:SEMIRING', 'sddnnf:SEMIRING'}}, found {}.".format(dm[0])
    if dm[0] == "fuzzy" or dm[0] == "sddnnf":
        assert len(dm) == 2, "Sum-product circuit automata ({}) take exactly 1 hyper-parameter (semiring), found {}.".format(dm[0], dm[1:])
        assert dm[1] in ["prob", "logprob"], "The supported semirings for sum-product circuits are {{'prob', 'logprob'}}, found {}.".format(dm[1])
        opts["dfa_module"] = {"type": dm[0], "semiring": dm[1]}
    elif dm[0] == "mlp" or dm[0] == "gru":
        assert len(dm) == 2, \
            "Neural {} automaton takes exactly 1 hyper-parameter (number of hidden neurons for next state prediction), found {}.".format(dm[0], dm[1:])
        assert int(dm[1], 10) > 1, \
            "The number of hidden neurons in the automaton module must be an integer >= 1, found {}.".format(dm[1])
        opts["dfa_module"] = {"type": dm[0], "hidden_n": int(dm[1], 10)}
    elif dm[0] == "scallop":
        assert len(dm) == 4, \
            "The Scallop automaton module takes exactly 3 hyper-parameters (provenance semiring, top-k value for training, top-k value for inference), found {}.".format(dm[1:])
        assert dm[1] in valid_provenances, \
            "The supported provenance semirings are {{{}}}, found {}.".format(", ".join(["'{}'".format(v) for v in valid_provenances]), dm[1])
        assert int(dm[2], 10) >= 1 and int(dm[3], 10) >= 1, \
            "Top-k values for Scallop must be positive integers, found train_k={}, test_k={}".format(dm[2], dm[3])
        opts["dfa_module"] = {"type": "scallop", "provenance": dm[1], "train_k": int(dm[2], 10), "test_k": int(dm[3], 10)}

    # # Check whether annotations exist. The actual data will be checked later.
    annotations_path = "{}/{}".format(opts["prefix_path"], opts["annotations_path"])
    assert os.path.exists(annotations_path), \
        "Annotation folder {} does not exist. Have you built the benchmark?".format(annotations_path)

    task_dir = "{}/{}".format(annotations_path, opts["task"])
    assert os.path.exists(task_dir), "Task folder {} does not exist.".format(task_dir)

    for split in ["train", "val", "test"]:
        assert os.path.exists("{}/sequence_{}.csv".format(task_dir, split)), \
            "Annotation file sequence_{}.csv does not exist.".format(split)
    for file in ["mappings", "automaton", "domain_values"]:
        assert os.path.exists("{}/{}.yml".format(task_dir, file)), "File {}.yml does not exist.".format(file)

    return opts

def prune_hyperparameters(opts, arg_parser):
    """
    Check whether irrelevant hyper-parameters have a non-default value. This allows to prune grid search space by
    skipping combinations of already-tried experiments.
    :param opts: Dictionary of hyper-parameters.
    :param arg_parser: argparse.ArgumentParser.
    :return: True if this experiment can be performed, False if it should be skipped.
    """

    ok = True

    if opts["teacher_forcing"] and opts["detach_previous_state"] != arg_parser.get_default("detach_previous_state"):
        ok = False
        print("Warning: With teacher forcing, detach_previous_state has no effect.")

    loss_weights = sum([v for k, v in opts.items() if k.endswith("_lambda")])
    if loss_weights <= 0:
        ok = False
        print("Warning: At least one loss function must be enabled.")

    if opts["calibrate"] != arg_parser.get_default("calibrate") and opts["constraint_module"]["type"] == "mlp" and \
            opts["dfa_module"]["type"] in ["mlp", "gru"]:
        ok = False
        print("Warning: Probability calibration is enabled only if at least one of constraint module or dfa module is symbolic.")

    if opts["scallop_e2e"] and opts["dfa_module"]["type"] != arg_parser.get_default("dfa_module").split(":")[0]:
        ok = False
        opts["scallop_e2e"] = False  # Reverting to allow continuation in case of --abort_irrelevant = False.
        print("Warning: dfa module is incompatible with --scallop_e2e=True. Ignoring the latter.")


    if opts["scallop_e2e"] and opts["constraint_module"]["type"] != "scallop":
        ok = False
        opts["scallop_e2e"] = False # Reverting to allow continuation in case of --abort_irrelevant = False.
        print("Warning: Constraint module is incompatible with --scallop_e2e=True. Ignoring the latter.")

    if opts["use_label_oracle"] and (opts["supervision_lambda"] != arg_parser.get_default("supervision_lambda") or
        opts["pretraining_epochs"] != arg_parser.get_default("pretraining_epochs")):
        ok = False
        opts["epochs"] += opts["pretraining_epochs"]
        opts["pretraining_epochs"] = 0
        print("Warning: When using the label oracle, supervision_lambda and pretraining_epochs have no effect.")


    if opts["use_constraint_oracle"] and opts["constraint_lambda"] != arg_parser.get_default("constraint_lambda"):
        ok = False
        print("Warning: When using the constraint oracle, constraint_lambda has no effect.")

    return ok


def domains_to_class_ids(task_opts):
    """
    Map domain values to unique IDs, no matter how streams are multiplexed in the task.
    :param task_opts: Dictionary of task options.
    :return: Nested dictionary: stream -> class name -> class id.
    """
    streams = {}
    for v in task_opts["mappings"].values():
        for var, stream in v["streams"].items():
            dir = stream["direction"]
            name = stream["name"]

            if var not in streams:
                streams[var] = set()
            streams[var].add("{}_{}".format(("var" if dir == "input" else "out"), name))


    classes = {}
    for v in task_opts["domain_values"].values():
        for var, domain in v.items():
            for stream in streams[var]:
                if stream not in classes:
                    classes[stream] = set()

                classes[stream].update(domain)


    for k, v in classes.items():
        classes[k] = {str(vv): i for i, vv in enumerate(sorted(v))}

    return classes

def sympy_to_scallop(constraint):
        out = constraint.replace("~", "not ").replace("&", ",").replace("|", " or ")
        out = re.sub(r"(p_\d+)", r"\1()", out)
        return out

class Watchdog:
    def __init__(self, function, heartbeat, *args, **kwargs):
        self.queue = multiprocessing.Queue()
        self.heartbeat = heartbeat

        self.fn_proc = multiprocessing.Process(target=self._fn_wrapper(function), args=args, kwargs=kwargs, daemon=True)

    def _fn_wrapper(self, function):
        def f(*args, **kwargs):
            for x in function(*args, **kwargs):
                self.queue.put(x)
        return f

    def listen(self, timeout):
        finished = False
        time = 0
        history = [] # Keep a local copy of the training history updated incrementally, to have some data to report in case of crashing/timeouts.
        return_tags = set()
        _, _, _ = self.queue.get(block=True)
        while not finished:
            if not self.fn_proc.is_alive():
                return_tags.add("Crashed")
                break

            try:
                phase, epoch_stats, tags = self.queue.get(block=True, timeout=self.heartbeat)
                time = 0

                if phase == "epoch":
                    history.append(epoch_stats) # Update partial history.
                else:  # Reached the end of the training loop.
                    finished = True
                    history = epoch_stats # Replace the entire history at the end of training, since it completed successfully.

                return_tags.update(tags)

            except queue.Empty:
                pass # In case of heartbeat timeout, there is still a chance of being within the epoch timeout.
            except KeyboardInterrupt:
                return_tags.add("User abort")
                break

            time += self.heartbeat

            if timeout > 0 and time >= timeout:
                return_tags.add("Timeout") # If the epoch timeout has expired, give up.
                break

        return history, return_tags

    def __enter__(self):
        self.fn_proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fn_proc.is_alive():
            self.fn_proc.kill()
            self.fn_proc.join()
            self.fn_proc.close()

        return True