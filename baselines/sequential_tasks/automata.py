import torch
import torch.nn.functional as F
from arpeggio.cleanpeg import ParserPEG
import arpeggio
import scallopy
from utils import sympy_to_scallop

class DFAMLP(torch.nn.Module):
    def __init__(self, num_neurons, propositions, automaton):
        super().__init__()

        num_states = len(automaton["states"])
        self.accepting = automaton["accepting"]
        self.non_accepting = sorted(set(automaton["states"]).difference(automaton["accepting"]))

        self.ns_predictor = torch.nn.Sequential(
            torch.nn.Linear(num_states + len(propositions), num_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(num_neurons, num_states)
        )

    def forward(self, log_s0, s0, constraints):
        prev_state = s0

        input_tensor = torch.cat([prev_state] + [constraints[p].unsqueeze(1) for p in sorted(constraints.keys())], dim=1)
        next_state = self.ns_predictor(input_tensor)

        sm_ns = F.softmax(next_state, dim=1)
        a = torch.sum(sm_ns[:, self.accepting], dim=1)
        b = (1.0 - torch.sum(sm_ns[:, self.non_accepting], dim=1))
        # XOR probability of being in an accepting state but not in a non-accepting one.
        # accepted = a also works, but it does not backpropagate gradients through non-accepting states.
        # Terminating on an accepting or non-accepting state are mutually exclusive events, so this is a well defined probability.
        accepted = a + b - a * b

        return F.log_softmax(next_state, dim=1), F.softmax(next_state, dim=1), accepted


class DFAGRU(torch.nn.Module):
    def __init__(self, num_neurons, propositions, automaton):
        super().__init__()
        num_states = len(automaton["states"])
        self.accepting = automaton["accepting"]
        self.non_accepting = sorted(set(automaton["states"]).difference(automaton["accepting"]))

        self.ns_predictor = torch.nn.GRUCell(len(propositions), num_neurons)

        self.encoder = torch.nn.Linear(num_states, num_neurons)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(num_neurons, num_states)
        )

    def forward(self, log_s0, s0, constraints):
        input_tensor = torch.stack([constraints[p] for p in sorted(constraints.keys())], dim=1)

        prev_state = s0

        enc_s0 = self.encoder(prev_state)
        enc_s1 = self.ns_predictor(input_tensor, enc_s0)
        next_state = self.decoder(enc_s1)

        sm_ns = F.softmax(next_state, dim=1)
        a = torch.sum(sm_ns[:, self.accepting], dim=1)
        b = (1.0 - torch.sum(sm_ns[:, self.non_accepting], dim=1))
        accepted = a + b - a * b

        return F.log_softmax(next_state, dim=1), F.softmax(next_state, dim=1), accepted

class DFAScallop(torch.nn.Module):
    def __init__(self, propositions, automaton, provenance, train_k, test_k, eps):
        super().__init__()

        self.propositions = propositions
        self.states = automaton["states"]
        self.eps = eps

        program = \
"""type prev_state(u8)
type next_state(u8)
type transition(bound u8, u8)
rel next_state(s1) = prev_state(s0), transition(s0, s1)

"""
        for p in self.propositions:
            program += "type {}()\n".format(p)
        program += "\n"

        for a in automaton["accepting"]:
            program += "rel accepting() = next_state({})\n".format(a)
        program += "\n"

        for s0, v in automaton.items():
            if isinstance(s0, int):
                for s1, trans in v.items():
                    if trans == "True":
                        program += "rel transition = {{({}, {})}}\n".format(s0, s1)
                    else:
                        program += "rel transition({},{}) = {}\n".format(s0, s1, sympy_to_scallop(trans))

        input_mappings = {p: () for p in self.propositions}
        input_mappings["prev_state"] = self.states

        output_mappings = {"next_state": self.states, "accepting": ()}


        self.net = scallopy.Module(
            program=program,
            input_mappings=input_mappings,
            output_mappings=output_mappings,
            provenance=provenance,
            train_k=train_k,
            test_k=test_k,
            retain_graph=True,
            #early_discard=False,
        )


    def forward(self, log_s0, s0, constraints):
        nesy_in = {p: v for p, v in constraints.items()}
        nesy_in["prev_state"] = s0

        nesy_out = self.net(**nesy_in)

        unnormalized_next_state = nesy_out["next_state"]
        next_state = unnormalized_next_state / (torch.sum(unnormalized_next_state, dim=1, keepdim=True).detach() + self.eps)

        return torch.log(next_state + self.eps), next_state, nesy_out["accepting"]


# For ease of implementation, the arithmetic circuit baselines are implemented by parsing the string corresponding to a negated-normal-form boolean formula.
# Note that this implementation is *NOT* probabilistically sound, as it suffers from the disjoint sum problem.
# The or of two independent events is Prob(a) + Prob(b).
# The or of two overlapping events is Prob(a) + Prob(b) - Prob(a) * Prob(b), but only if a and b are leaf nodes.
# This approach cannot discriminate them, so it will over-estimate probability mass, for example:
# (a&b) | (a&c) will be unfold as Prob(a)*Prob(b) + Prob(a) * Prob(c) - Prob(a)*Prob(b)*Prob(a)*Prob(c)  (counting Prob(a) twice)
# In a fuzzy setting (i.e., when this circuit is used as t-norm), it can still provide useful gradients, however, for
# correct probabilistic semantics, it should be preceded by a knowledge compilation step to guarantee every variable is independent.

# Kudos to <https://github.com/nmanginas> for pointing this out.

# Important: even replacing the fuzzy circuit with a correct probabilistic one, we would still be unable to discriminate
# non-independent events, because constraint semantics would be shadowed by constraint names. For example,
# if p_0 = "X+Y=Z" and p_1 = "X-Y=Z", their probabilities are not independent, as the case (X=Z, Y=0) affects both constraints.
# Since the DFA module is not aware of constraint semantics, but only of their aliases, it can only consider p_0 and p_1 independent.
# For this reason we have the additional --scallop_e2e flag, for experiments where constraint semantics is propagated to
# next state prediction as well.


class NNFVisitor(arpeggio.PTNodeVisitor):
    def __init__(self, semiring, weights=None):
        super().__init__()
        self.semiring = semiring
        self.weights = weights if weights is not None else {}
        self.weights["True"] = self.semiring.one()

    def set_weights(self, weights):
        self.weights = weights
        self.weights["True"] = self.semiring.one() * torch.ones_like(weights[sorted(weights.keys())[0]])

    def visit_prop(self, node, children):
        return self.weights[node.value]

    def visit_literal(self, node, children):
        if children[0] == "~":
            return self.semiring.neg(children[-1])
        else:
            return children[0]

    def visit_and_expr(self, node, children):
        term = self.semiring.one() * self.weights["True"]

        for c in children:
            term = self.semiring.mul(term, c)
        return term

    def visit_or_expr(self, node, children):
        term = torch.nan_to_num(self.semiring.zero() * self.weights["True"], nan=self.semiring.zero())
        for c in children:
            term = self.semiring.add(term, c)
        return term


class ProbSemiring():

    def zero(self):
        return 0.0

    def one(self):
        return 1.0

    def neg(self, t):
        return 1.0 - t

    def add(self, t1, t2):
        return t1 + t2

    def mul(self, t1, t2):
        return t1 * t2

class LogSemiring:
    def zero(self):
        return -torch.inf

    def one(self):
        return 0.0

    def neg(self, t):
        return torch.log1p(-torch.exp(t))

    def add(self, t1, t2):
        return torch.logaddexp(t1, t2)

    def mul(self, t1, t2):
        return t1 + t2

class DFAProb(torch.nn.Module):
    def __init__(self, automaton, semiring, eps, neginf):
        super().__init__()

        assert semiring in ["prob", "logprob"], "Unknown semiring {}. It must be either of {{'prob', 'logprob'}}".format(semiring)

        self.automaton = automaton
        self.eps = eps
        self.neginf = neginf

        nnf_grammar = """
                    prop = r'p_\d+' / "True"
                    literal = ("~"? prop) / "(" or_expr ")"

                    and_expr = literal ("&" literal)*
                    or_expr = and_expr ("|" and_expr)*
                    formula = or_expr EOF
                """

        self.apply_log = (semiring == "logprob")
        self.parser = ParserPEG(nnf_grammar, "formula")
        self.visitor = NNFVisitor(LogSemiring() if self.apply_log else ProbSemiring())
        self.accepting = automaton["accepting"]
        self.non_accepting = sorted(set(automaton["states"]).difference(automaton["accepting"]))

        self.dfa = {k: v for k, v in automaton.items() if isinstance(k, int)}


    def _normalize_probs(self, probs):
        return probs / torch.sum(probs, dim=1, keepdim=True).detach()

    def _normalize_logs(self, probs):
        return probs - torch.logsumexp(probs, dim=1, keepdim=True).detach()

    def _compute_transition_probs(self, prev_probs, constr_probs):
        # Simply visit every state of the DFA and compute the probability of each transition, given the probabilities of the previous state and of atomic constraints being satisfied.
        out = [[self.visitor.semiring.zero() * torch.ones(prev_probs.size(0), device=prev_probs.device)] for _ in range(len(self.dfa.keys()))]

        self.visitor.set_weights(constr_probs)

        for s1 in self.dfa.keys():
            for s0 in self.dfa.keys():
                if s1 in self.dfa[s0]:
                    parse_tree = self.parser.parse(self.dfa[s0][s1])
                    next_state_prob = arpeggio.visit_parse_tree(parse_tree, self.visitor)

                    out[s1].append(self.visitor.semiring.mul(prev_probs[:, s0], next_state_prob))

        if self.apply_log:
            ret = torch.stack([torch.logsumexp(torch.stack(o, dim=1), dim=1) for o in out], dim=1)
        else:
            ret = torch.stack([torch.sum(torch.stack(o, dim=1), dim=1) for o in out], dim=1)


        return ret

    def forward(self, log_s0, s0, constraints):
        if self.apply_log:
            # Clamping both inputs and outputs is necessary to avoid: 1) nan gradient, 2) underflows with float32.
            prev_state = torch.clamp(log_s0, self.neginf, -self.eps)
            clamped_constraints = {p: torch.log(torch.clamp(v, self.eps, 1.0 - self.eps)) for p, v in constraints.items()}
            unnormalized_ns = self._compute_transition_probs(prev_state, clamped_constraints)
            log_next_state = torch.clamp(self._normalize_logs(unnormalized_ns), self.neginf, -self.eps)
            next_state = torch.clamp(torch.exp(log_next_state), self.eps, 1.0 - self.eps)
        else:
            prev_state = s0

            #clamped_constraints = {p:  for p, v in constraints.items()}
            unnormalized_ns = self._compute_transition_probs(prev_state, constraints) # clamped_constraints)
            # Clamping required to avoid NaN gradients.
            next_state = torch.clamp(self._normalize_probs(unnormalized_ns), self.eps, 1.0 - self.eps)
            log_next_state = torch.log(next_state)

        a = torch.sum(next_state[:, self.accepting], dim=1)
        b = (1.0 - torch.sum(next_state[:,self.non_accepting], dim=1))
        accepted = a + b - a * b

        return log_next_state, next_state, accepted