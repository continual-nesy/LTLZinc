import re

import sympy.logic.boolalg
from flloat.parser.ltlf import LTLfParser
import numpy as np

class LTLAutomaton:
    """
    Automaton class. It builds a deterministic finite state automaton from an LTLf formula and samples traces from it,
    while keeping track of data streams.
    """
    def __init__(self, ltl, pred_dict, streams, avoid_policy, truncate_on_sink, constraint_sampling, rng, silent=True):
        self.avoid_policy = avoid_policy
        self.rng = rng
        self.streams = streams
        self.truncate_on_sink = truncate_on_sink
        self.constraint_sampling = constraint_sampling
        self.silent = silent

        assert self.constraint_sampling in [True, False, "positive", "negative"]

        self.inverse_streams = {}

        for k, v in streams.items():
            vv = v[1:] if (v.startswith("+") or v.startswith("-")) else v
            if vv not in self.inverse_streams:
                self.inverse_streams[vv] = []
            self.inverse_streams[vv].append(k)

        if not self.silent:
            print("Building formula...")
        self._build_formula(ltl, pred_dict)
        self._build_automaton()

    def _build_formula(self, ltl, pred_dict):
        """
        Build a vocabulary of constraint substitutions and a propositional version of the LTLf formula.
        :param ltl: The LTLf formula.
        :param pred_dict: The dictionary of constraint definitions.
        """
        ltl = re.sub(r'\s*,\s*', ',', ltl)

        extracted_preds = set(re.findall(r'[a-z][0-9a-z]*\([A-Z,\s]+\)', ltl))
        for k in pred_dict.keys():
            if k in ltl:
                extracted_preds.add(k)

        extracted_preds = sorted(extracted_preds)


        assert len(extracted_preds) > 0, \
            "At least one predicate must be present in the ltl specification, found {}.".format(ltl)

        i = 0
        self.mapping = {}
        self.formula = ltl

        encountered_preds = set()

        for pred in extracted_preds:
            if "(" not in pred:
                if pred in ltl:
                    encountered_preds.add(pred)
            else:
                name = pred.split("(")[0]
                arity = pred.count(",") + 1
                tmp = re.sub(r'\s', '', pred).split("(")[1].split(")")[0].split(",")

                found = False
                for k, v in pred_dict.items():
                    pred_name = k.split("(")[0]
                    pred_arity = k.count(",") + 1

                    if name == pred_name and arity == pred_arity:
                        found = True
                        subs = {v["terms"][j]: tmp[j] for j in range(len(tmp))}
                        chans = {}
                        for k in tmp:
                            if self.streams[k].startswith("-"):
                                chans[k] = {"name": self.streams[k][1:], "direction": "output"}
                            elif self.streams[k].startswith("+"):
                                chans[k] = {"name": self.streams[k][1:], "direction": "input"}
                            else:
                                chans[k] = {"name": self.streams[k], "direction": "input"}

                        propositional_name = "p_{}".format(i)
                        self.mapping[propositional_name] = {
                            "predicate": pred_name, "substitutions": subs,
                            "streams": chans,
                            "constraint": v["constraint"],
                            "orphan": False
                        }
                        self.formula = self.formula.replace(pred, propositional_name)
                        encountered_preds.add(name)
                assert found, "Predicate {} appears in formula, but not in predicate list.".format(pred)
                i += 1

        # Adding orphan predicates (appearing in predicate list, but not in the ltl formula).
        for p, v in pred_dict.items():
            name = p.split("(")[0]
            if v is None:  # Orphan predicate defined in minizinc_prefix.
                self.mapping[name] = {
                    "predicate": name, # There is no ambiguity since arity = 0.
                    "substitutions": None,
                    "streams": None,
                    "constraint": None,
                    "orphan": name not in encountered_preds
                }
            else:
                if name not in encountered_preds:
                    propositional_name = "p_{}".format(i) # Arity >0: there can be ambiguity after grounding. Using an incremental index.
                    tmp = re.sub(r'\s', '', p).split("(")[1].split(")")[0].split(",")
                    subs = {v["terms"][j]: tmp[j] for j in range(len(tmp))}

                    chans = {}
                    for k in tmp:
                        if self.streams[k].startswith("-"):
                            chans[k] = {"name": self.streams[k][1:], "direction": "output"}
                        elif self.streams[k].startswith("+"):
                            chans[k] = {"name": self.streams[k][1:], "direction": "input"}
                        else:
                            chans[k] = {"name": self.streams[k], "direction": "input"}

                    self.mapping[propositional_name] = {
                        "predicate": name,
                        "substitutions": subs,
                        "streams": chans,
                        "constraint": v["constraint"],
                        "orphan": True
                    }

                    i += 1



    def _build_automaton(self):
        """
        Build a DFA from a propositionalized LTLf formula.
        Then check whether the formula is unsatisfiable or a tautology
        (in either case, there would be nothing to learn from data).
        """
        parser = LTLfParser()

        if not self.silent:
            print("Converting to nnf...")
        nnf = parser(self.formula).to_nnf()
        if not self.silent:
            print("Building automaton...")
        self.dfa = nnf.to_automaton()
        self._identify_special_states()


        assert len(self.dfa.accepting_states) > 0, "The formula is unsatisfiable."

        immediate_accepting = self.dfa.accepting_states == set(self.dfa._transition_function[self.dfa.initial_state].keys())
        if immediate_accepting:
            tautology = True
            for acc in self.dfa.accepting_states:
                tautology &= str(self.dfa._transition_function[acc][acc]) == "True"
            assert not tautology, "The formula is a tautology, there is no useful temporal information."

    #
    def export_automaton(self):
        """
        Export the automaton in human-readable text format, along with the internal mapping linking variables,
        streams and propositional aliases.
        :return: Tuple (human-readable automaton, machine-readable automaton, internal mapping).
        """
        str_automaton = "States: {{{}}}\n".format(", ".join([str(k) for k in self.dfa._transition_function.keys()]))
        str_automaton += "Initial state: {}\n".format(self.dfa.initial_state)
        str_automaton += "Accepting states: {{{}}}\n".format(", ".join([str(k) for k in self.dfa.accepting_states]))
        str_automaton += "\nTransitions:\n"
        for src_state in sorted(self.dfa._transition_function.keys()):
            for dst_state in sorted(self.dfa._transition_function[src_state].keys()):
                str_automaton += "[{}->{}]\t{}\n".format(src_state, dst_state, self.dfa._transition_function[src_state][dst_state])

        dict_automaton = {
            "states": sorted(self.dfa._transition_function.keys()),
            "initial": self.dfa.initial_state, "accepting": sorted(self.dfa.accepting_states)
        }
        for s0, v in self.dfa._transition_function.items():
            dict_automaton[s0] = {}
            for s1, formula in v.items():
                dict_automaton[s0][s1] = str(formula)

        return str_automaton, dict_automaton, self.mapping

    # Compute the sampling probability for loop states.
    def _decay_probs(self, t, state_type):
        if self.avoid_policy[state_type][0] == "linear":
            return min(1.0, (t+1) * self.avoid_policy[state_type][1])
        else:
            return 1. - np.exp(-(t+1) * self.avoid_policy[state_type][1])

    def _identify_special_states(self):
        self.loop_states = set() #  Loops with guard != True.
        self.accepting_sinks = set() # Loops with guard == True and in accepting.
        self.rejecting_sinks = set() # Loops with guard == True and not in accepting.

        self.constraints = set()

        for cs, v in  self.dfa._transition_function.items():
            for ns, f in v.items():
                if cs == ns:
                    if str(f) != "True":
                        self.loop_states.add(cs)
                    elif cs in self.dfa.accepting_states:
                        self.accepting_sinks.add(cs)
                    else:
                        self.rejecting_sinks.add(cs)

    def _compute_probs(self, succs, stats):
        probs = []
        for s in succs:
            if s in self.loop_states:
                probs.append(self._decay_probs(stats["self_loops"], "self_loops"))
                stats["self_loops"] += 1
            elif s in self.accepting_sinks:
                probs.append(self._decay_probs(stats["accepting_sinks"], "accepting_sinks"))
                stats["accepting_sinks"] += 1
            elif s in self.rejecting_sinks:
                probs.append(self._decay_probs(stats["rejecting_sinks"], "rejecting_sinks"))
                stats["rejecting_sinks"] += 1
            else:
                probs.append(1.0)

        # In the end probabilities are normalized.
        if np.sum(probs) > 0:
            probs = np.array(probs) / np.sum(probs)
        else:  # All probabilities are 0 (e.g. because every next_state is absorbing and the policy gives 0 probability): force them to 1/n.
            probs = np.ones_like(probs) / len(probs)

        return probs

    def _ldfs(self, cur_state, target_type, stats, max_depth=0, pos_constraints=None, neg_constraints=None):
        assert target_type in ["accepting", "rejecting", "any"]
        ok = target_type == "any"
        ok |= target_type == "accepting" and cur_state in self.dfa.accepting_states
        ok |= target_type == "rejecting" and cur_state not in self.dfa.accepting_states


        if self.constraint_sampling != False:
            if pos_constraints is None:
                pos_constraints = set()
            if neg_constraints is None:
                neg_constraints = set()

            if self.constraint_sampling == True or self.constraint_sampling == "positive":
                ok &= not pos_constraints.isdisjoint([str(k) for k in self.mapping.keys() if not self.mapping[k]["orphan"]])
            if self.constraint_sampling == True or self.constraint_sampling == "negative":
                ok &= not neg_constraints.isdisjoint([str(k) for k in self.mapping.keys() if not self.mapping[k]["orphan"]])

        if max_depth == 0:
            return [cur_state], ok
        elif cur_state in self.accepting_sinks:
            postfix_length = 1 if self.truncate_on_sink["accepting"] else max_depth
            return [cur_state] * postfix_length, ok
        elif cur_state in self.rejecting_sinks:
            postfix_length = 1 if self.truncate_on_sink["rejecting"] else max_depth
            return [cur_state] * postfix_length, ok
        else:  # Recursive case.
            succs = sorted(self.dfa._transition_function[cur_state].keys())
            while len(succs) > 0: # Either returns inside the loop or continues until no successor is found.
                probs = self._compute_probs(succs, stats)
                next_state = self.rng.choice(succs, p=probs)
                if self.constraint_sampling != False:
                    formula = self.dfa._transition_function[cur_state][next_state]
                    atoms = formula.atoms()

                    for a in atoms:
                        if formula.count(sympy.logic.boolalg.Not(a)) > 0:
                            neg_constraints.add(str(a))
                        elif formula.count(a) > 0:
                            pos_constraints.add(str(a))

                tail, ok = self._ldfs(next_state, target_type, stats, max_depth - 1, pos_constraints, neg_constraints)

                if ok:
                    return [cur_state] + tail, ok
                else:
                    succs.remove(next_state)
            if len(succs) == 0:
                return [], False # Search failed. No path from cur_state satisfies target_type.


    def generate_sequence(self, steps, target_type="any"):
        """
        Randomly traverse the automaton to generate a trace.
        :param steps: The number of steps to walk.
        :return:
        """
        assert isinstance(steps, int) or len(steps) == 2, \
            "Steps must be an int or a tuple (min, max), found {}.".format(steps)
        assert target_type in ["accepting", "rejecting", "any"], \
            "Sequence type must be one of ['accepting', 'rejecting', 'any'], found {}.".format(target_type)

        min_steps = steps[0]
        ok = False
        while not ok and min_steps <= steps[1]:
            seq_len = self.rng.integers(min_steps, steps[1] + 1)

            stats = {"self_loops": 0, "accepting_sinks": 0,  "rejecting_sinks": 0}
            state_trace, ok = self._ldfs(self.dfa.initial_state, target_type, stats, max_depth=seq_len)

            if not ok: # If there is no trace of length seq_len which satisfies the target type, increase the sequence length and retry.
                min_steps = seq_len + 1

        # If a maximum-length sequence (steps[1]) still cannot satisfy the target type, it is an unrecoverable error.
        # Note that for incremental mode, steps[0] = steps[1], therefore the loop will terminate in one cycle.
        assert ok, "No trace in the automaton ends in a {} state.".format(target_type)

        formula_trace = []
        transition_trace = []
        stream_trace = []

        for i in range(len(state_trace) - 1):
            cs = state_trace[i]
            ns = state_trace[i + 1]
            t = self.dfa._transition_function[cs][ns]

            formula_trace.append(t)
            transition_trace.append("{}->{}".format(cs, ns))

            tmp = {}
            for symb in t.free_symbols:
                if self.mapping[str(symb)]["streams"] is not None:
                    for k, v in self.mapping[str(symb)]["streams"].items():
                        if k in tmp:
                            assert tmp[k] == v["name"], \
                                "Found conflict for variable {}: {} != {}.".format(k, v["name"], tmp[k])
                        else:
                            assert v["name"] not in tmp.values(), \
                                "Found conflict for variable {}: {} is already in {}".format(k, v["name"], tmp)

                        tmp[k] = v["name"]

            # Fill don't care streams with random values.
            for missing_stream in sorted(set(self.inverse_streams.keys()).difference(tmp.values())):
                v = self.rng.choice(self.inverse_streams[missing_stream])
                tmp[v] = "({})".format(missing_stream)

            stream_trace.append(tmp)


        return formula_trace, transition_trace, state_trace[-1] in self.dfa.accepting_states, stream_trace


    def generate_sequences(self, steps, num_sequences, positive_only=False, balance=False):
        """
        Generate multiple sequences.
        :param steps:  Length of target sequences.
        :param num_sequences: Number of sequences to generate.
        :param balance: If True, try to guarantee a balanced dataset. Give up after self.patience attempts (after that point sequences will no longer be balanced).
        :return:
        """

        formula_traces = []
        transition_traces = []
        label_traces = []
        stream_traces = []

        for i in range(num_sequences):
            if positive_only:
                f, t, l, s = self.generate_sequence(steps, "accepting")
            elif balance:
                f, t, l, s = self.generate_sequence(steps, "accepting" if i % 2 == 0 else "rejecting")
            else:
                f, t, l, s = self.generate_sequence(steps, "any")

            formula_traces.append(f)
            transition_traces.append(t)
            label_traces.append(l)
            stream_traces.append(s)

        idx = list(range(num_sequences))
        self.rng.shuffle(idx)
        return [formula_traces[i] for i in idx], [transition_traces[i] for i in idx], [label_traces[i] for i in idx], [stream_traces[i] for i in idx]