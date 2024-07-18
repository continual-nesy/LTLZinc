import re
from flloat.parser.ltlf import LTLfParser
import numpy as np

class LTLAutomaton:
    """
    Automaton class. It builds a deterministic finite state automaton from an LTLf formula and samples traces from it,
    while keeping track of data streams.
    """
    def __init__(self, ltl, pred_dict, streams, avoid_absorbing, truncate_on_absorbing, rng, patience):
        self.avoid_absorbing = avoid_absorbing
        self.rng = rng
        self.patience = patience
        self.streams = streams
        self.truncate_on_absorbing = truncate_on_absorbing

        self.inverse_streams = {}

        for k, v in streams.items():
            vv = v[1:] if (v.startswith("+") or v.startswith("-")) else v
            if vv not in self.inverse_streams:
                self.inverse_streams[vv] = []
            self.inverse_streams[vv].append(k)

        self._build_formula(ltl, pred_dict)
        self._build_automaton()

    def _build_formula(self, ltl, pred_dict):
        """
        Build a vocabulary of constraint substitutions and a propositional version of the LTLf formula.
        :param ltl: The LTLf formula.
        :param pred_dict: The dictionary of constraint definitions.
        """
        ltl = re.sub(r'\s*,\s*', ',', ltl)

        extracted_preds = sorted(set(re.findall(r'[a-z][0-9a-z]*\([A-Z,\s]+\)', ltl)))

        assert len(extracted_preds) > 0, \
            "At least one predicate must be present in the ltl specification, found {}.".format(ltl)

        i = 0
        self.mapping = {}
        self.formula = ltl

        for pred in extracted_preds:
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
                    #chans = {k: self.streams[k] for k in tmp}
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
                        "predicate": k, "substitutions": subs,
                        "streams": chans,
                        "constraint": v["constraint"]
                    }
                    self.formula = self.formula.replace(pred, propositional_name)
            assert found, "Predicate {} appears in formula, but not in predicate list.".format(pred)
            i += 1

    def _build_automaton(self):
        """
        Build a DFA from a propositionalized LTLf formula.
        Then check whether the formula is unsatisfiable or a tautology
        (in either case, there would be nothing to learn from data).
        """
        parser = LTLfParser()

        self.dfa = parser(self.formula).to_nnf().to_automaton()

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

    # Compute the sampling probability for absorbing states.
    def _get_probs(self, t, state_type):
        if self.avoid_absorbing[state_type][0] == "linear":
            return min(1.0, t * self.avoid_absorbing[state_type][1])
        else:
            return 1. - np.exp(-t * self.avoid_absorbing[state_type][1])

    def _identify_absorbing_states(self, cur_state):
        """
        Perform look-ahead search for absorbing next states.
        :param cur_state: The current state.
        :return: Tuple (set of next states which are absorbing and accepting, set of next states which are absorbing and non-accepting).
        """
        next_acc_abs = set()
        next_rej_abs = set()

        for next_state in self.dfa._transition_function[cur_state].keys():
            for next_next, formula in self.dfa._transition_function[next_state].items(): # The transition of the successor.
                if str(formula) == "True":
                    if next_next in self.dfa.accepting_states:
                        next_acc_abs.add(next_state)
                    else:
                        next_rej_abs.add(next_state)

        return next_acc_abs, next_rej_abs

    def generate_sequence(self, steps):
        """
        Randomly traverse the automaton to generate a trace.
        :param steps: The number of steps to walk.
        :return:
        """
        assert isinstance(steps, int) or len(steps) == 2, \
            "Steps must be an int or a tuple (min, max), found {}.".format(steps)
        cur_state = self.dfa.initial_state
        formula_trace = []
        transition_trace = []
        cur_streams = []
        seq_len = self.rng.integers(steps[0], steps[1] + 1)

        abs_acc = 0 # increased every time an absorbing accepting state is a possible next state
        abs_rej = 0 # increased every time an absorbing rejecting state is a possible next state

        for i in range(seq_len):
            next_acc_abs, next_rej_abs = self._identify_absorbing_states(cur_state)
            next_non_abs = {s for s in self.dfa._transition_function[cur_state].keys() if s not in next_acc_abs and s not in next_rej_abs}

            if len(next_acc_abs) > 0:
                abs_acc += 1
            if len(next_rej_abs) > 0:
                abs_rej += 1

            next_states = []
            probs = []
            acc_mass = self._get_probs(abs_acc, "accepting")

            # Biases next state selection, based on absorbing states avoidance policy.
            for s in sorted(next_acc_abs):
                next_states.append(s)
                probs.append(acc_mass)

            rej_mass = self._get_probs(abs_rej, "rejecting")
            for s in sorted(next_rej_abs):
                next_states.append(s)
                probs.append(rej_mass)

            for s in sorted(next_non_abs):
                next_states.append(s)
                probs.append(1.0)

            # In the end probabilities are normalized.
            if np.sum(probs) > 0:
                probs = np.array(probs) / np.sum(probs)
            else: # All probabilities are 0 (e.g. because every next_state is absorbing and the policy gives 0 probability): force them to 1/n.
                probs = np.ones_like(probs) / len(probs)

            next_state = self.rng.choice(next_states, p=probs)
            transition = self.dfa._transition_function[cur_state][next_state]

            formula_trace.append(transition)
            transition_trace.append("{}->{}".format(cur_state, next_state))

            tmp = {}
            for symb in transition.free_symbols:
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

            cur_state = next_state
            cur_streams.append(tmp)

            # Entered an absorbing state. Prematurely truncating the sequence if necessary.
            if str(transition) == "True" and self.truncate_on_absorbing["accepting" if next_state in next_acc_abs else "rejecting"]:
                break

        return formula_trace, transition_trace, cur_state in self.dfa.accepting_states, cur_streams

    def generate_accepting_sequence(self, steps):
        """
        # Rejection sampling of a single accepting sequence. Give up after self.patience failed attempts.
        :param steps: Length of the target sequence.
        :return:
        """
        for attempts in range(self.patience):
            formula_trace, transition_trace, accepting, cur_streams = self.generate_sequence(steps)
            if accepting:
                break

        assert attempts < self.patience - 1, "Ran out of patience while generating an accepting sequence." # This is an error because the method should always return accepting sequences.

        return formula_trace, transition_trace, cur_streams

    def generate_accepting_sequences(self, steps, num_sequences):
        """
        # Generate multiple accepting sequences.
        :param steps: Length of target sequences.
        :param num_sequences: Number of sequences to generate.
        :return:
        """
        traces = []
        transitions = []
        cur_streams = []
        for _ in range(num_sequences):
            trace, transition_trace, cur_chan = self.generate_accepting_sequence(steps)
            traces.append(trace)
            transitions.append(transition_trace)
            cur_streams.append(cur_chan)

        return traces, transitions, cur_streams

    def generate_sequences(self, steps, num_sequences, balance=False):
        """
        Generate multiple sequences.
        :param steps:  Length of target sequences.
        :param num_sequences: Number of sequences to generate.
        :param balance: If True, try to guarantee a balanced dataset. Give up after self.patience attempts (after that point sequences will no longer be balanced).
        :return:
        """
        pos_traces = 0
        neg_traces = 0
        traces = []
        transitions = []
        labels = []
        stream_seq = []

        attempts = 0
        while len(traces) < num_sequences:

            trace, transition_trace, accepting, cur_streams = self.generate_sequence(steps)
            # Allow a difference of 1 if the requested number of sequences is odd.
            if balance and abs(pos_traces - neg_traces) > num_sequences % 2:
                select_positive = pos_traces < neg_traces
                while attempts < self.patience and accepting != select_positive:
                    trace, transition_trace, accepting, cur_streams = self.generate_sequence(steps)
                    attempts += 1

            pos_traces += 1 if accepting else 0
            neg_traces += 1 if not accepting else 0
            traces.append(trace)
            transitions.append(transition_trace)
            labels.append(accepting)
            stream_seq.append(cur_streams)
        return traces, transitions, labels, stream_seq