import pandas as pd

from automaton import LTLAutomaton
from sampler import StateSampler

import tqdm.auto as tqdm

import sympy
import os
import glob
import abc

class AbstractGenerator(abc.ABC):
    """
    Base class for automaton initialization and image sampling.
    """
    def __init__(self, config, rng):
        self.rng = rng

        self.length = config["length"]
        self.formula = config["formula"]
        self.predicates = config["predicates"]
        self.domains = config["domains"]
        self.types = config["types"]
        self.streams = config["streams"]
        self.patience = config["patience"]

        self.splits = config["splits"]

        self.dfa = LTLAutomaton(config["formula"], config["predicates"], config["streams"],
                                config["avoid_states"], config["truncate_on_absorbing"], self.rng,
                                config["patience"])

        self.sampler = StateSampler(config["domains"], config["types"], self.dfa.mapping, self.rng,
                                    config["minizinc_prefix"])

        self.vars = sorted(self.sampler.vars)
        self.props = sorted(self.sampler.props)

    @abc.abstractmethod
    def get_dataset(self, **kwargs):
        raise NotImplementedError()

    def _get_random_img(self, split, var, label):
        """
        Get a random image from the folder corresponding to the given assignment.
        :param split: The dataset split to use.
        :param var: The variable name.
        :param label: The variable value.
        :return: A random filepath for an image of the requested variable.
        """
        assert split in self.splits.keys(), "Split must be one of {}, found {}.".format(self.splits.keys(), split)

        path = "{}/{}".format(self.splits[split]["path"], self.types[self.domains[var][split]][label])
        assert os.path.exists(path), "Directory {} not found.".format(path)

        files = []
        for ext in ["png", "jpeg", "jpg", "bmp"]:
            files.extend(glob.glob("{}/*.{}".format(path, ext)))

        return self.rng.choice(files)

class SequenceGenerator(AbstractGenerator):
    """
    Generator for sequence based tasks.
    """
    def __init__(self, config, rng, positive_only=False, silent=True):
        super().__init__(config, rng)
        self.positive_only = positive_only
        self.silent = silent

    def get_dataset(self, split, balance=False):
        """
        Generate a pandas DataFrame of sequences.
        :param split: The target split.
        :param balance: If True reject sequences which would make the dataset imbalanced.
        :return: A pandas.DataFrame.
        """
        assert split in self.splits.keys(), "Split must be one of {}, found {}.".format(self.splits.keys(), split)

        n = self.splits[split]["samples"]

        dataset = []
        if self.positive_only:
            traces, transition_traces, streams = self.dfa.generate_accepting_sequences(self.length, n)
        else:
            traces, transition_traces, labels, streams = self.dfa.generate_sequences(self.length, n, balance=balance)

        for i in tqdm.trange(n, position=1, desc="seqs", leave=False, disable=self.silent):
            for j in range(len(traces[i])): # Always maximum length.
                tmp = {k: None for k in self.props}

                assignments, props = self.sampler.solve(traces[i][j], solutions=1, split=split)

                for k, v in assignments[0].items():
                    if k in streams[i][j]:
                        tmp_var = streams[i][j][k]
                        if tmp_var.startswith("("):
                            tmp.update({"var_{}".format(tmp_var[1:-1]): "({})".format(v)})
                            tmp.update({"img_{}".format(tmp_var[1:-1]): self._get_random_img(split, k, v)})
                        else:
                            tmp.update({"var_{}".format(tmp_var): str(v)})
                            tmp.update({"img_{}".format(tmp_var): self._get_random_img(split, k, v)})


                tmp.update(props[0])
                tmp.update({"transition": transition_traces[i][j]})
                tmp.update({"time": j})



                if self.positive_only:
                    tmp.update({"seq_id": i, "accepted": 1})
                else:
                    tmp.update({"seq_id": i, "accepted": 1 if labels[i] else 0})
                dataset.append(tmp)

        df = pd.DataFrame(dataset)

        cols = ["seq_id", "time", "accepted", "transition"] + [c for c in sorted(df.columns) if c.startswith("img_")]
        cols += [c for c in sorted(df.columns) if c.startswith("var_")]
        cols += [c for c in sorted(df.columns) if c.startswith("p_")]

        return df[cols]

class TaskGenerator(AbstractGenerator):
    """
    Generator for incremental tasks. The automaton trace is always accepting and always truncated before the last non-absorbing state.
    """
    def __init__(self, config, rng, silent=True):
        super().__init__(config,rng)

        self.length = config["length"][0]
        self.silent = silent

        self.curriculum, self.transition_trace, self.seq_streams = self.dfa.generate_accepting_sequence([self.length, self.length])
        self.curriculum = [task for task in self.curriculum if not isinstance(task, sympy.logic.boolalg.BooleanTrue)] # Automatically remove absorbing states.

    def get_dataset(self, split):
        """
        Generate a pandas DataFrame of tasks.
        :param split: The target split.
        :return: A pandas.DataFrame.
        """
        assert split in self.splits.keys(), "Split must be one of {}, found {}.".format(self.splits.keys(), split)

        n = self.splits[split]["samples"]


        seq = []
        for i, task in tqdm.tqdm(enumerate(self.curriculum), position=1, desc="tasks", leave=False, disable=self.silent):
            assignments, props = self.sampler.solve(task, solutions=n)
            for j in tqdm.trange(n, position=2, desc="seqs", leave=False, disable=self.silent):
                tmp = {"task_id": i, "transition": self.transition_trace[i]}
                tmp.update({k: None for k in self.props})
                for k, v in assignments[j].items():
                    if k in self.seq_streams[i]:
                        tmp_var = self.seq_streams[i][k]
                        if tmp_var.startswith("("):
                            tmp.update({"var_{}".format(tmp_var[1:-1]): "({})".format(v)})
                            tmp.update({"img_{}".format(tmp_var[1:-1]): self._get_random_img(split, k, v)})
                        else:
                            tmp.update({"var_{}".format(tmp_var): str(v)})
                            tmp.update({"img_{}".format(tmp_var): self._get_random_img(split, k, v)})

                tmp.update(props[j])
                tmp.update({"seq_id": j})

                seq.append(tmp)

        df = pd.DataFrame(seq)

        cols = ["task_id", "seq_id", "transition"] + [c for c in sorted(df.columns) if c.startswith("img_")]
        cols += [c for c in sorted(df.columns) if c.startswith("var_")]
        cols += [c for c in sorted(df.columns) if c.startswith("p_")]

        return df[cols]
