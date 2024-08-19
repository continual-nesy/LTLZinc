import torch
import torch.nn.functional as F
from arpeggio.cleanpeg import ParserPEG
import arpeggio
import nnf
import scallopy
from utils import sympy_to_scallop

class DFAMLP(torch.nn.Module):
    def __init__(self, num_neurons, propositions, automaton, calibrate, eps):
        super().__init__()

        num_states = len(automaton["states"])
        self.accepting = automaton["accepting"]
        self.non_accepting = sorted(set(automaton["states"]).difference(automaton["accepting"]))

        self.calibrate = calibrate
        self.eps = eps
        self.temperature_logit = torch.nn.Parameter(torch.ones(1, requires_grad=True))


        self.ns_predictor = torch.nn.Sequential(
            torch.nn.Linear(num_states + len(propositions), num_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(num_neurons, num_states)
        )

    def forward(self, log_s0, s0, constraints):
        prev_state = s0

        input_tensor = torch.cat([prev_state] + [constraints[p].unsqueeze(1) for p in sorted(constraints.keys())], dim=1)
        next_state = self.ns_predictor(input_tensor)

        if self.calibrate:
            t = F.relu(self.temperature_logit) + self.eps
        else:
            t = 1.0

        sm_ns = F.softmax(next_state / t, dim=1)
        accepted = torch.sum(sm_ns[:, self.accepting], dim=1)

        return F.log_softmax(next_state / t, dim=1), sm_ns, accepted


class DFAGRU(torch.nn.Module):
    def __init__(self, num_neurons, propositions, automaton, calibrate, eps):
        super().__init__()
        num_states = len(automaton["states"])
        self.accepting = automaton["accepting"]
        self.non_accepting = sorted(set(automaton["states"]).difference(automaton["accepting"]))

        self.ns_predictor = torch.nn.GRUCell(len(propositions), num_neurons)

        self.encoder = torch.nn.Linear(num_states, num_neurons)
        self.decoder = torch.nn.Linear(num_neurons, num_states)

        self.calibrate = calibrate
        self.eps = eps
        self.temperature_logit = torch.nn.Parameter(torch.ones(1, requires_grad=True))

    def forward(self, log_s0, s0, constraints):
        input_tensor = torch.stack([constraints[p] for p in sorted(constraints.keys())], dim=1)

        prev_state = s0

        enc_s0 = self.encoder(prev_state)
        enc_s1 = self.ns_predictor(input_tensor, enc_s0)
        next_state = self.decoder(enc_s1)

        if self.calibrate:
            t = F.relu(self.temperature_logit) + self.eps
        else:
            t = 1.0

        sm_ns = F.softmax(next_state / t, dim=1)
        accepted = torch.sum(sm_ns[:, self.accepting], dim=1)

        return F.log_softmax(next_state / t, dim=1), F.softmax(next_state / t, dim=1), accepted

class DFAScallop(torch.nn.Module):
    def __init__(self, propositions, automaton, provenance, train_k, test_k, calibrate, eps):
        super().__init__()

        self.propositions = propositions
        self.states = automaton["states"]
        self.eps = eps
        self.calibrate = calibrate

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

        self.temperature_logit = torch.nn.Parameter(torch.ones(1, requires_grad=True))


    def forward(self, log_s0, s0, constraints):
        nesy_in = {p: v for p, v in constraints.items()}
        nesy_in["prev_state"] = s0

        nesy_out = self.net(**nesy_in)

        unnormalized_next_state = nesy_out["next_state"]

        if self.calibrate:
            t = F.relu(self.temperature_logit) + self.eps
        else:
            t = 1.0

        next_state = F.softmax(torch.log(unnormalized_next_state) / t, dim=1)
        log_next_state = F.log_softmax(torch.log(unnormalized_next_state) / t, dim=1)

        return log_next_state, next_state, nesy_out["accepting"]

class BoolVisitor(arpeggio.PTNodeVisitor):
    def visit_atom(self, node, children):
        if node.value == "True":
            return nnf.true
        elif node.value == "False":
            return nnf.false
        else:
            return nnf.Var(node.value)

    def visit_factor(self, node, children):
        if children[0] == "~":
            return children[-1].negate()
        else:
            return children[0]

    def visit_and_expr(self, node, children):
        term = nnf.true

        for c in children:
            term &= c
        return term

    def visit_or_expr(self, node, children):
        term = nnf.false
        for c in children:
            term |= c
        return term


class SumModule(torch.nn.Module):
    def __init__(self, children, semiring, eps, neginf):
        super().__init__()
        self.children_modules = torch.nn.ModuleList(children)
        self.eps = eps
        self.neginf = neginf
        self.semiring = semiring

    def forward(self, weights):
        if self.semiring == "prob":
            return torch.sum(torch.stack([c(weights) for c in self.children_modules], dim=-1), dim=-1)
        else:
            return torch.logsumexp(torch.stack([c(weights) for c in self.children_modules], dim=-1), dim=-1)

class ProdModule(torch.nn.Module):
    def __init__(self, children, semiring, eps, neginf):
        super().__init__()
        self.children_modules = torch.nn.ModuleList(children)
        self.eps = eps
        self.neginf = neginf
        self.semiring = semiring

    def forward(self, weights):
        if self.semiring == "prob":
            return torch.cumprod(torch.stack([c(weights) for c in self.children_modules], dim=-1), dim=-1)[:,-1]
        else:
            return torch.sum(torch.stack([c(weights) for c in self.children_modules], dim=-1), dim=-1)

class NodeModule(torch.nn.Module):
    def __init__(self, node, semiring, eps, neginf):
        super().__init__()

        self.eps = eps
        self.neginf = neginf
        self.semiring = semiring

        if isinstance(node, nnf.Var):
            self.negated = not node.true
            self.name = node.name
        elif node == nnf.true:
            self.negated = False
            self.name = "True"
        elif node == nnf.false:
            self.negated = True
            self.name = "False"

    def forward(self, weights):
        dummy = weights[sorted(weights.keys())[0]]
        if self.name == "True":
            if self.semiring == "prob":
                return torch.ones(dummy.shape, device=dummy.device, requires_grad=False)
            else:
                return torch.ones(dummy.shape, device=dummy.device, requires_grad=False) * -self.eps
        elif self.name == "False":
            if self.semiring == "prob":
                return torch.zeros(dummy.shape, device=dummy.device, requires_grad=False)
            else:
                return torch.ones(dummy.shape, device=dummy.device, requires_grad=False) * self.neginf
        elif self.negated:
            if self.semiring == "prob":
                return 1.0 - weights[self.name]
            else:
                return torch.log1p(-torch.exp(weights[self.name]))
        else:
            return weights[self.name]

class DFAProb(torch.nn.Module):
    def __init__(self, automaton, type, semiring, calibrate, eps, neginf):
        super().__init__()

        assert type in ["fuzzy", "sddnnf"], "Unknown sum-product circuit type {}. It must be either of {{'fuzzy', 'sddnnf'}}".format(type)
        assert semiring in ["prob", "logprob"], "Unknown semiring {}. It must be either of {{'prob', 'logprob'}}".format(semiring)

        self.automaton = automaton
        self.type = type
        self.semiring = semiring
        self.calibrate = calibrate
        self.eps = eps
        self.neginf = neginf

        self.temperature_logit = torch.nn.Parameter(torch.ones(1, requires_grad=True))

        bool_grammar = """
            atom = r'[a-zA-Z][a-zA-Z0-9_]*'
            factor = "~"? (atom / "(" or_expr ")")
            and_expr = factor ("&" factor)*
            or_expr = and_expr ("|" and_expr)*
            formula = or_expr+ EOF
        """

        self.parser = ParserPEG(bool_grammar, "formula")
        self.visitor = BoolVisitor()
        self.accepting = automaton["accepting"]
        self.non_accepting = sorted(set(automaton["states"]).difference(automaton["accepting"]))

        dfa = {k: v for k, v in automaton.items() if isinstance(k, int)}
        transposed_dfa = {k: {k2: "False" for k2 in dfa.keys()} for k in dfa.keys()}
        for ps, v in dfa.items():
            for ns, constr in v.items():
                transposed_dfa[ns][ps] = constr

        formulas = {}
        for ns, v in transposed_dfa.items():
            formulas[ns] = nnf.false
            for ps, constr in v.items():
                parse_tree = self.parser.parse(constr)
                constr_formula = arpeggio.visit_parse_tree(parse_tree, self.visitor)

                formulas[ns] |= nnf.Var("ps_{}".format(ps)) & constr_formula

            formulas[ns] = formulas[ns].simplify()

            if self.type == "sddnnf": # This baseline is inspired by the yet-unpublished work of <https://github.com/nmanginas>.
                formulas[ns] = nnf.dsharp.compile(formulas[ns].to_CNF(), smooth=True).forget_aux().simplify()

        self.dfa = torch.nn.ModuleDict({str(k): self._visit_formula(v) for k, v in formulas.items()})

    def _visit_formula(self, node):
        if node.leaf():
            return NodeModule(node, self.semiring, self.eps, self.neginf)
        else:
            tmp = [self._visit_formula(c) for c in node.children]
            if isinstance(node, nnf.And):
                return ProdModule(tmp, self.semiring, self.eps, self.neginf)
            else:
                return SumModule(tmp, self.semiring, self.eps, self.neginf)

    # def _normalize_probs(self, probs):
    #     return probs / torch.sum(probs, dim=1, keepdim=True).detach()
    #
    # def _normalize_logs(self, logprobs):
    #     return logprobs - torch.logsumexp(logprobs, dim=1, keepdim=True).detach()

    def forward(self, log_s0, s0, constraints):
        if self.calibrate:
            t = F.relu(self.temperature_logit) + self.eps
        else:
            t = 1.0


        if self.semiring == "prob":
            weights = {"ps_{}".format(i): torch.clamp(s0[:,i], self.eps, 1.0 - self.eps) for i in range(s0.size(1))}
            weights.update(constraints)

            unnormalized_ns = torch.clamp(torch.stack([self.dfa[k](weights) for k in sorted(self.dfa.keys())], dim= -1), self.eps, 1.0 - self.eps)
            #next_state = torch.clamp(self._normalize_probs(unnormalized_ns), self.eps, 1.0 - self.eps)
            #log_next_state = torch.log(next_state)


            next_state = F.softmax(torch.log(unnormalized_ns) / t, dim=-1) # In probability space, thus we need log.
            log_next_state = F.log_softmax(torch.log(unnormalized_ns) / t, dim=-1) # In probability space, thus we need log.

        else:
            weights = {"ps_{}".format(i): torch.clamp(log_s0, self.neginf, -self.eps)[:,i] for i in range(log_s0.size(1))}
            weights.update({p: torch.log(torch.clamp(v, self.eps, 1.0 - self.eps)) for p, v in constraints.items()})
            unnormalized_ns = torch.clamp(torch.stack([self.dfa[k](weights) for k in sorted(self.dfa.keys())], dim=-1), self.neginf, -self.eps)

            #log_next_state = torch.clamp(self._normalize_logs(unnormalized_ns), self.neginf, -self.eps)
            #next_state = torch.clamp(torch.exp(log_next_state), self.eps, 1.0 - self.eps)

            next_state = F.softmax(unnormalized_ns / t, dim=-1) # Already in logit space.
            log_next_state = F.log_softmax(unnormalized_ns / t, dim=-1) # Already in logit space.

        accepted = torch.sum(next_state[:, self.accepting], dim=-1)

        return log_next_state, next_state, accepted
