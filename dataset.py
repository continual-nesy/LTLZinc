import pandas as pd

from automaton import LTLAutomaton
from sampler import StateSampler

import tqdm.auto as tqdm

import sympy
import os
import glob
import abc
import numpy as np

import minizinc

class AbstractGenerator(abc.ABC):
    """
    Base class for automaton initialization and image sampling.
    """
    def __init__(self, config, rng, silent=True):
        self.rng = rng

        self.silent = silent
        self.length = config["length"]
        self.formula = config["formula"]
        self.predicates = config["predicates"]
        self.domains = config["domains"]
        self.types = config["types"]
        self.streams = config["streams"]
        self.constraint_sampling = config["guarantee_constraint_sampling"]
        self.guarantee_mode = config["guarantee_mode"]

        self.splits = config["splits"]

        self.dfa = LTLAutomaton(config["formula"], config["predicates"], config["streams"],
                                config["avoid_states"], config["truncate_on_sink"],
                                self.constraint_sampling, self.rng, silent=self.silent)

        self.orphans = {sympy.Symbol(k) for k, v in self.dfa.mapping.items() if v["orphan"]}
        self.orphans.update({~o for o in self.orphans})

        self.sampler = StateSampler(config["domains"], config["types"], self.dfa.mapping, self.rng, config["minizinc_prefix"])

        self.vars = sorted(self.sampler.vars)
        self.props = sorted(self.sampler.props)

        if not self.silent:
            print("Caching solutions and removing unsolvable transitions...")

        unsolvable_transitions = []
        for s0, v in self.dfa.dfa._transition_function.items():
            # Cache formulas without orphans.
            for s1, formula in v.items():
                try:
                    self.sampler.solve(formula)
                except RuntimeError as e:
                    print(e)
                    unsolvable_transitions.append((s0, s1))

        for s0, s1 in unsolvable_transitions:
            del self.dfa.dfa._transition_function[s0][s1]
            if len(self.dfa.dfa._transition_function[s0]) == 0:
                del self.dfa.dfa._transition_function[s0]

        # Cache formulas augmented with ONE orphan.
        if self.constraint_sampling != False:
            self.incompatible_orphans = {}
            for s0, v in self.dfa.dfa._transition_function.items():
                for s1, formula in v.items():
                    if str(formula) not in self.incompatible_orphans:
                        self.incompatible_orphans[str(formula)] = set()
                    for o in self.orphans:
                        try:
                            self.sampler.solve(formula & o)
                        except RuntimeError as e:
                            print(e)
                            self.incompatible_orphans[str(formula)].add(o)

    @abc.abstractmethod
    def get_dataset(self, **kwargs):
        raise NotImplementedError()

    def _augment_trace(self, trace, solver="gecode"):
        if self.constraint_sampling == False:
            if not self.silent:
                print("Trace augmentation is not required.")
            return trace
        elif self.constraint_sampling == "positive":
            target = {o for o in self.orphans if isinstance(o, sympy.Symbol)}
        elif self.constraint_sampling == "negative":
            target = {o for o in self.orphans if isinstance(o, sympy.logic.Not)}
        else:
            target = self.orphans

        sorted_target = sorted(target, key=lambda t: str(t))
        program = "include \"count.mzn\";\narray[0..{}] of var 0..{}: trace;\n".format(len(trace) - 1, len(target))

        for i, t in enumerate(trace):
            compatible_orphans = sorted([str(sorted_target.index(o) + 1) for o in target.difference(self.incompatible_orphans[str(t)])]) #  +1  since index 0 is reserved for no augmentation.

            program += "constraint trace[{}] in {{0, {}}};\n".format(i, ", ".join(compatible_orphans))

        if self.guarantee_mode == "strict_f":
            program += "\nconstraint forall(i in 1..{})(count(trace, i) >= 1);\nsolve satisfy; ".format(len(target))
        elif self.guarantee_mode == "strict_g":
            program += "\nconstraint forall(i in 1..{})(count(trace, i) >= 1);\nconstraint count(trace, 0) = 0;\nsolve satisfy;".format(len(target))
        elif self.guarantee_mode == "best_f":
            program += "\nsolve maximize sum(i in 1..{})(bool2int(count(trace, i) >= 1));".format(len(target))
        elif self.guarantee_mode == "best_g":
            program += "\nconstraint forall(i in 1..{})(count(trace, i) >= 1);\nsolve minimize count(trace, 0);".format(len(target))

        mzn_solver = minizinc.Solver.lookup(solver)
        model = minizinc.Model()
        model.add_string(program)
        instance = minizinc.Instance(mzn_solver, model)
        result = instance.solve() # All solutions would be too slow to be computed for each trace (they cannot be cached), picking only the first one. Solve maximize/minimize with gecode would only yield one anyway.
        if not result.status.has_solution():
            raise RuntimeError("The following problem has no solution:\n{}".format(model._code_fragments))

        augmentations = result.solution.__dict__["trace"]

        aug_trace = []
        for i, t in enumerate(trace):
            if augmentations[i] == 0:
                aug_trace.append(t)
            else:
                aug_trace.append(t & sorted_target[augmentations[i] - 1])

        return aug_trace

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
        super().__init__(config, rng, silent)
        self.positive_only = positive_only

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
        if not self.silent:
            print("Generating sequences...")
        if self.positive_only:
            traces, transition_traces, _, streams = self.dfa.generate_sequences(self.length, n, positive_only=True)
        else:
            traces, transition_traces, labels, streams = self.dfa.generate_sequences(self.length, n, balance=balance)

        if not self.silent:
            print("Augmenting traces...")

        traces = [self._augment_trace(t) for t in traces]

        if not self.silent:
            print("Sampling assignments...")

        orphan_props = {str(o) for o in self.orphans if isinstance(o, sympy.Symbol)}

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

                tmp_props = {k: (v if k not in orphan_props else "[{}]".format(v.replace("(", "").replace(")", ""))) for k, v in props[0].items()}

                tmp.update(tmp_props)
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
    Generator for incremental tasks. The automaton trace is always accepting and always truncated before the last non-sink state.
    """
    def __init__(self, config, rng, silent=True):
        super().__init__(config, rng, silent)

        self.length = config["length"][0]

        self.guarantee_ratio = config["incremental_guarantee_ratio"]

        if not self.silent:
            print("Generating sequence...")
        self.curriculum, self.transition_trace, _, self.seq_streams = self.dfa.generate_sequence([self.length, self.length],"accepting")
        self.curriculum = [task for task in self.curriculum if not isinstance(task, sympy.logic.boolalg.BooleanTrue)] # Automatically remove sink states.

        if not self.silent:
            print("Augmenting sequence with guarantees...")

        self.augmented_curriculum = self._augment_trace(self.curriculum)

        if not self.silent:
            print("Completed sequence generation.")

    def get_dataset(self, split):
        """
        Generate a pandas DataFrame of tasks.
        :param split: The target split.
        :return: A pandas.DataFrame.
        """
        assert split in self.splits.keys(), "Split must be one of {}, found {}.".format(self.splits.keys(), split)

        n = self.splits[split]["samples"]

        orphan_props = {str(o) for o in self.orphans if isinstance(o, sympy.Symbol)}

        seq = []
        for i, task in tqdm.tqdm(enumerate(self.curriculum), position=1, desc="tasks", leave=False, disable=self.silent):
            if not self.silent:
                print("Solving instances for task {}/{}...".format(i+1, len(self.curriculum)))

            if self.guarantee_ratio == "once":
                n_aug = 1
            else:
                n_aug = int(np.ceil(n * self.guarantee_ratio))

            # Solve n - n_aug unguaranteed samples and n_aug guaranteed ones, then merge the two datasets and shuffle.
            assignments, props = self.sampler.solve(task, solutions=n - n_aug)
            if n_aug > 0:
                aug_assignments, aug_props = self.sampler.solve(self.augmented_curriculum[i], solutions=n_aug)
                assignments.extend(aug_assignments)
                props.extend(aug_props)

                idx = self.rng.permutation(len(assignments))
                assignments = [assignments[i] for i in idx]
                props = [props[i] for i in idx]


            if not self.silent:
                print("Sampling sequences for task {}/{}...".format(i + 1, len(self.curriculum)))
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

                # Orphan propositions are marked as key: "[value]", while normal propositions as key: "value" or key: "(value)".
                tmp_props = {k: (v if k not in orphan_props else "[{}]".format(v.replace("(", "").replace(")", ""))) for k, v in props[j].items()}
                tmp.update(tmp_props)
                tmp.update({"seq_id": j})

                seq.append(tmp)

        df = pd.DataFrame(seq)

        cols = ["task_id", "seq_id", "transition"] + [c for c in sorted(df.columns) if c.startswith("img_")]
        cols += [c for c in sorted(df.columns) if c.startswith("var_")]
        cols += [c for c in sorted(df.columns) if c.startswith("p_")]

        return df[cols]
