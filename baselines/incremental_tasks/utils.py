import random
import time
import numpy as np
import torch
import hashlib
import argparse
import os

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
    arg_parser.add_argument('--task_seed', help="Task seed to use (default: '12345')", type=ArgNumber(int), default=12345)

    # Training parameters.
    arg_parser.add_argument("--lr", help="Learning rate (positive: SGD, negative: Adam; default: -1e-4)",
                            type=ArgNumber(float), default=-1e-4)
    arg_parser.add_argument("--epochs", help="Training epochs (default: 10)", type=ArgNumber(int, min_val=1),
                            default=10)
    arg_parser.add_argument('--shuffle',
                            help="Shuffle training samples before each epoch (default: True)",
                            type=ArgBoolean(), default=True)
    arg_parser.add_argument("--batch_size", help="Batch size (default: 32)", type=ArgNumber(int, min_val=1), default=32)
    arg_parser.add_argument("--buffer_batch_size", help="Batch size for experience replay (default: 4)", type=ArgNumber(int, min_val=0),
                            default=4)
    arg_parser.add_argument("--eval_batch_size", help="Batch size for evaluation (default: 128)", type=ArgNumber(int, min_val=1), default=128)
    arg_parser.add_argument("--supervision_lambda", help="Weight for direct supervision (default: 1.0)",
                            type=ArgNumber(float, min_val=0.0), default=1.0)
    arg_parser.add_argument("--replay_lambda", help="Weight for experience replay loss (default: 1.0)",
                            type=ArgNumber(float, min_val=0.0), default=1.0)
    arg_parser.add_argument("--distil_lambda", help="Weight for distillation loss (default: 1.0)",
                            type=ArgNumber(float, min_val=0.0), default=1.0)
    arg_parser.add_argument('--grad_clipping', help="Norm for gradient clipping, disable if 0.0 (default: 0.0)",
                            type=ArgNumber(float, min_val=0.0), default=0.0)
    arg_parser.add_argument('--train_on_irrelevant',
                            help="Apply the loss also on irrelevant annotations (default: False)", type=ArgBoolean(),
                            default=False)
    arg_parser.add_argument("--buffer_size", help="Replay buffer size (default: 200)", type=ArgNumber(int, min_val=1), default=200)
    arg_parser.add_argument("--buffer_type", help="Type of replay buffer in {'none', 'ring', 'reservoir'} (default: 'none')",
                            type=str, choices=["none", "ring", "reservoir"], default="none")
    arg_parser.add_argument("--buffer_loss",
                            help="Type of experience replay loss in {'ce', 'gem'} (default: 'ce')",
                            type=str, choices=["ce", "gem"], default="ce")
    arg_parser.add_argument("--distillation_source",
                            help="Data source for distillation in {'none', 'batch', 'buffer', 'both'} (distillation is disabled if 'none'; default: 'none')",
                            type=str, choices=["none", "batch", "buffer", "both"], default="none")
    arg_parser.add_argument("--modular_net", help="Allocate a different layer for each concept (default: False)",
                            type=ArgBoolean(), default=False)

    # Model parameters.
    arg_parser.add_argument('--backbone_module',
                            help="Perceptual backbone in {'squeezenet11', 'shufflenetv2x05', 'googlenet', 'densenet121'} (default: 'squeezenet11')",
                            type=str, default="cnn", choices=["squeezenet11", "shufflenetv2x05", "googlenet", "densenet121"])
    arg_parser.add_argument('--pretrained_weights',
                            help="If true, initialize backbone with ImageNet weights (default: False)",
                            type=ArgBoolean(), default=False)
    arg_parser.add_argument("--emb_size", help="Embedding size of the backbone (default: 128)", type=ArgNumber(int, min_val=1), default=128)
    arg_parser.add_argument("--hidden_size", help="Embedding size of each of the hidden layers (default: 32)", type=ArgNumber(int, min_val=1), default=32)
    arg_parser.add_argument('--knowledge_availability',
                            help="Knowledge availability in {'none', 'predicates', 'states'} (default: 'none')",
                            type=str, choices=["none", "predicates", "states"], default="none")


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
                            help="Amount of output (0: no output, 1: task summary, 2: epoch summary, 3: metrics eval summary, 4: full; default: 1)",
                            type=int, choices=[0, 1, 2, 3, 4], default=1)
    arg_parser.add_argument("--wandb_project", help="Use W&B, this is the project name (default: None)", type=str,
                            default=None)
    arg_parser.add_argument('--eval_train',
                            help="If true, compute metrics also for the training set (slow; default: False)",
                            type=ArgBoolean(), default=False)

    # Magic numbers.
    arg_parser.add_argument('--eps',
                            help="Epsilon for numerical stability (default: 1e-3)",
                            type=ArgNumber(float, min_val=0), default=1e-3)
    arg_parser.add_argument('--neginf',
                            help="Negative infinity for numerical stability (default: -1e10)",
                            type=ArgNumber(float, max_val=0), default=-1e10)
    arg_parser.add_argument('--vanishing_threshold',
                            help="Threshold triggering a vanishing gradient tag (default: 1e-5)",
                            type=ArgNumber(float, min_val=0), default=1e-5)
    arg_parser.add_argument('--exploding_threshold',
                            help="Threshold triggering an exploding gradient tag (default: 1e7)",
                            type=ArgNumber(float, min_val=0), default=1e7)
    arg_parser.add_argument('--std_threshold',
                            help="Threshold triggering an high gradient standard deviation tag (default: 200.0)",
                            type=ArgNumber(float, min_val=0), default=200.0)
    arg_parser.add_argument('--heartbeat',
                            help="Heartbeat for the watchdog timer in seconds (default: 10)",
                            type=ArgNumber(int, min_val=1), default=10)

    return arg_parser

def get_unhashable_opts():
    """
    Explicit list of arguments which should NOT be used for the unique identifier computation.
    :return: The list of excluded arguments.
    """
    return ["wandb_project", "save", "task_seed", "seed", "device", "annotations_path", "data_path", "eps", "neginf",
            "overfitting_threshold", "vanishing_threshold", "exploding_threshold", "std_threshold", "rnd_threshold",
            "mp_threshold", "heartbeat", "epoch_timeout", "verbose", "eval_train", "eval_batch_size"]

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

    # # Check whether annotations exist. The actual data will be checked later.
    annotations_path = "{}/{}".format(opts["prefix_path"], opts["annotations_path"])
    assert os.path.exists(annotations_path), \
        "Annotation folder {} does not exist. Have you built the benchmark?".format(annotations_path)

    task_dir = "{}/{}".format(annotations_path, opts["task"])
    assert os.path.exists(task_dir), "Task folder {} does not exist.".format(task_dir)

    seed_dir = "{}/{}".format(task_dir, opts["task_seed"])
    assert os.path.exists(seed_dir), "Seed {} for task {} does not exist.".format(opts["task_seed"], opts["task"])

    for split in ["train", "val", "test"]:
        assert os.path.exists("{}/incremental_{}.csv".format(seed_dir, split)), \
            "Annotation file incremental_{}.csv does not exist.".format(split)
    for file in ["mappings", "automaton", "domain_values", "../target_metrics"]:
        assert os.path.exists("{}/{}.yml".format(seed_dir, file)), "File {}.yml does not exist.".format(file)

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


    loss_weights = sum([v for k, v in opts.items() if k.endswith("_lambda")])
    if loss_weights <= 0:
        ok = False
        print("Warning: At least one loss function must be enabled.")

    if opts["buffer_type"] == "none" and opts["distillation_source"] not in ["none", "batch"]:
        ok = False
        print("Warning: Without experience replay, distillation is only possible within the batch.")
        if opts["distillation_source"] == "buffer":
            opts["distillation_source"] = "none"
        else:
            opts["distillation_source"] = "batch"

    if opts["buffer_type"] == "none" and opts["buffer_loss"] != arg_parser.get_default("buffer_loss"):
        ok = False
        print("Warning: Without a replay buffer, buffer loss has no effect.")

    if opts["buffer_type"] == "none" and opts["replay_lambda"] != arg_parser.get_default("replay_lambda"):
        ok = False
        print("Warning: Without a replay buffer, replay_lambda has no effect.")

    if opts["distillation_source"] == "none" and opts["distil_lambda"] != arg_parser.get_default("distil_lambda"):
        ok = False
        print("Warning: Without knowledge distillation, distil_lambda has no effect.")

    if opts["buffer_batch_size"] == 0 and opts["buffer_type"] != "none" and opts["distillation_source"] in ["batch", "both"]:
        ok = False
        print("Warning: when using experience replay or buffer-based distillation, buffer_batch_size cannot  be zero. Resetting buffer batch to default value.")
        opts["buffer_batch_size"] = arg_parser.get_default("buffer_batch_size")
    if opts["buffer_size"] == 0 and opts["buffer_type"] != "none" and opts["distillation_source"] in ["batch", "both"]:
        ok = False
        print("Warning: when using experience replay or buffer-based distillation, buffer_size cannot  be zero. Resetting buffer batch to default value.")
        opts["buffer_size"] = arg_parser.get_default("buffer_size")

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
                phase, task_stats, tags = self.queue.get(block=True, timeout=self.heartbeat)
                time = 0

                if phase == "epoch":
                    pass # Nothing to do at the end of each epoch, except consuming from the queeue.
                elif phase == "task":
                    history.append(task_stats) # Update partial history.
                elif phase == "end":  # Reached the end of the training loop.
                    finished = True
                    history = task_stats # Replace the entire history at the end of training, since it completed successfully.

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