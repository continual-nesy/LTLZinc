import torch
import scallopy
import torch.nn.functional as F

from constraint_modules import ConstraintMLP, ConstraintScallop
from automata import DFAMLP, DFAGRU, DFAProb, DFAScallop
from backbones import backbone_factory
from utils import sympy_to_scallop

class SequenceClassifier(torch.nn.Module):
    """
    Generic classifier, it contains a perceptual component (self.backbone), a constraint predictor (self.constraints) and
    an automaton module (self.dfa).
    """
    def __init__(self, opts, task_opts, classes):
        super().__init__()

        self.backbone = torch.nn.ModuleDict(backbone_factory[opts["backbone_module"]][opts["task"]]())

        self.propositions = sorted(task_opts["mappings"].keys())
        num_classes = sum([len(v) for v in classes.values()])
        self.num_states = len(task_opts["automaton"]["states"])
        self.variables = sorted(classes.keys())

        program_path = "{}/{}/{}".format(opts["prefix_path"], opts["annotations_path"], opts["task"])

        # Constraint module can either be an MLP or a Scallop program.
        if opts["constraint_module"]["type"] == "mlp":
            self.constraints = ConstraintMLP(
                opts["constraint_module"]["neurons"],
                num_classes,
                self.propositions,
                opts["calibrate"],
                opts["eps"])
        else:
            self.constraints = ConstraintScallop(
                program_path,
                classes,
                self.propositions,
                opts["constraint_module"]["provenance"],
                opts["constraint_module"]["train_k"],
                opts["constraint_module"]["test_k"],
                opts["calibrate"],
                opts["eps"]
            )

        # DFA module can be: MLP, GRU, fuzzy automaton with probability semiring,
        # fuzzy automaton with log-probability semiring, or Scallop program.
        if opts["dfa_module"]["type"] == "mlp":
            self.dfa = DFAMLP(
                opts["dfa_module"]["hidden_n"],
                self.propositions,
                task_opts["automaton"],
                opts["calibrate"],
                opts["eps"]
            )
        elif opts["dfa_module"]["type"] == "gru":
            self.dfa = DFAGRU(
                opts["dfa_module"]["hidden_n"],
                self.propositions,
                task_opts["automaton"],
                opts["calibrate"],
                opts["eps"]
            )
        elif opts["dfa_module"]["type"] == "fuzzy" or opts["dfa_module"]["type"] == "sddnnf":
            self.dfa = DFAProb(task_opts["automaton"],
                               type=opts["dfa_module"]["type"],
                               semiring=opts["dfa_module"]["semiring"],
                               calibrate=opts["calibrate"],
                               eps=opts["eps"],
                               neginf=opts["neginf"]
            )
        elif opts["dfa_module"]["type"] == "scallop":
            self.dfa = DFAScallop(
                self.propositions,
                task_opts["automaton"],
                opts["dfa_module"]["provenance"],
                opts["dfa_module"]["train_k"],
                opts["dfa_module"]["test_k"],
                opts["calibrate"],
                opts["eps"]
            )


    def forward(self, log_s0, s0, imgs):
        img_labels = {v: self.backbone[v](imgs[i]) for i, v in enumerate(self.variables)}
        constraint_labels = self.constraints(img_labels)
        log_next_state, next_state, accepted = self.dfa(log_s0, s0, constraint_labels)

        return [img_labels[k] for k in sorted(img_labels.keys())], \
            [constraint_labels[p] for p in sorted(constraint_labels.keys())], \
            log_next_state, next_state, accepted


    def forward_sequence(self, imgs, mask, true_states=None, detach_prev_state=False):
        """
        Process a batch of sequences of different lengths by iterating in time_steps, instead of the naive approach
        in time_steps * batch_size. Formally the complexity is the same, but batching is delegated to torch for efficient
        handling.
        :param imgs: List of lists of images. The outermost dimension indexes variables, the innermost the sequence length.
        :param mask: torch.Tensor mask for each sequence of the batch.
        :param true_states: List of one-hot true states for each time step. If None disables teacher forcing.
        :param detach_prev_state: If True, detaches the gradient at each step, during backpropagation through time.
        :return: Tuple (List of variable predictions, List of constraint predictions, torch.Tensor of state traces, sequence label).
                 Categorical values are returned in log space, while scalar values are returned in [0-1].
        """
        shape = None
        for x in imgs:
            if shape is None:
                shape = x.shape
            else:
                assert x.shape == shape, "All images must have the same shape, found: {}.".format([y.shape for y in imgs])
        assert mask.size(0) == shape[0] and mask.size(1) == shape[1]  # (batch, timestep)

        batch_size, seq_len, c, w, h = shape

        assert true_states is None or true_states.shape == (batch_size, seq_len + 1)

        if true_states is not None:
            one_hot_true_states = F.one_hot(torch.where(true_states >= 0, true_states, 0), num_classes=self.num_states).to(torch.float32)
            log_one_hot_true_states = torch.log(one_hot_true_states)
        else:
            one_hot_true_states = None

        # Lists avoid overwriting variables (which would cause errors in the backward pass).
        log_states = [-torch.inf * torch.ones((batch_size, self.num_states), device=imgs[0].device)]
        states = [torch.zeros((batch_size, self.num_states), device=imgs[0].device)]
        vars = [[] for _ in self.variables]
        prop_labels = [[] for _ in self.propositions]

        for i in range(len(states)):
            log_states[i][:, 0] = 0.
            states[i][:, 0] = 1.  # One-hot initial state. Assuming initial state is always 0.

        label = torch.zeros(batch_size, device=imgs[0].device)
        for i in range(seq_len):
            non_masked_batch = mask[:,i].nonzero().squeeze(1)
            non_masked_batch_tuple = mask[:, i].nonzero(as_tuple=True)
            sliced_imgs = [x[non_masked_batch, i, :, :, :] for x in imgs]


            if non_masked_batch.size(0) > 0:
                if one_hot_true_states is None:
                    if detach_prev_state:
                        tmp_vars, tmp_constr, tmp_log_s1, tmp_s1, tmp_label = self.forward(log_states[i][non_masked_batch,:].detach(), states[i][non_masked_batch,:].detach(), sliced_imgs)
                    else:
                        tmp_vars, tmp_constr, tmp_log_s1,  tmp_s1, tmp_label = self.forward(log_states[i][non_masked_batch,:], states[i][non_masked_batch,:], sliced_imgs)
                else:  # Teacher forcing:
                    tmp_vars, tmp_constr, tmp_log_s1, tmp_s1, tmp_label = self.forward(log_one_hot_true_states[non_masked_batch, i, :].squeeze(1), one_hot_true_states[non_masked_batch, i, :].squeeze(1), sliced_imgs)

                log_states.append(log_states[-1].index_put(values=tmp_log_s1, indices=non_masked_batch_tuple))
                states.append(states[-1].index_put(values=tmp_s1, indices=non_masked_batch_tuple))
                for j, _ in enumerate(self.variables):
                    if len(vars[j]) > 0:
                        vars[j].append(vars[j][-1].index_put(values=tmp_vars[j], indices=non_masked_batch_tuple))
                    else:
                        vars[j].append(tmp_vars[j])

                for j, _ in enumerate(self.propositions):
                    if len(prop_labels[j]) > 0:
                        prop_labels[j].append(prop_labels[j][-1].index_put(values=tmp_constr[j], indices=non_masked_batch_tuple))
                    else:
                        prop_labels[j].append(tmp_constr[j])

                label = label.index_put(values=tmp_label, indices=non_masked_batch_tuple)
            else: # Every sequence in the batch has been truncated. No need to continue iterating, just copy old values.
                log_states.append(log_states[-1])
                states.append(states[-1])

                for j, _ in enumerate(self.variables):
                    vars[j].append(vars[j][-1])
                for j, _ in enumerate(self.propositions):
                    prop_labels[j].append(prop_labels[j][-1])

        return [torch.stack(v, dim=1) for v in vars], [torch.stack(p, dim=1) for p in prop_labels], torch.stack(log_states, dim=1), label


class ScallopE2E(SequenceClassifier):
    """
    End-to-End Scallop program. It computes jointly constraint and next state probabilities.
    Significantly slower than the two step approach, but it handles the case of non mutually exclusive constraints
    (e.g., p_0 = "z == x + y", p_1 = "z == x - y" which have in common solutions where y = 0, the two step approach cannot
    detect such cases, because the semantic of each constraint is hidden behind the unique identifiers p_0 and p_1).
    """
    def __init__(self, opts, task_opts, classes):
        torch.nn.Module.__init__(self)

        self.backbone = torch.nn.ModuleDict(backbone_factory[opts["backbone_module"]][opts["task"]]())

        self.propositions = sorted(task_opts["mappings"].keys())
        self.num_states = len(task_opts["automaton"]["states"])
        self.variables = sorted(classes.keys())
        self.eps = opts["eps"]

        program_path = "{}/{}/{}/scallop_constraints.scl".format(opts["prefix_path"], opts["annotations_path"], opts["task"])

        with open(program_path, "r") as file:
            program = file.read()

        # Extend the Scallop program with the DFA.
        program += \
"""\ntype prev_state(u8)
type next_state(u8)
type transition(bound u8, u8)
rel next_state(s1) = prev_state(s0), transition(s0, s1)

"""

        for a in task_opts["automaton"]["accepting"]:
            program += "rel accepting() = next_state({})\n".format(a)
        program += "\n"

        for s0, v in task_opts["automaton"].items():
            if isinstance(s0, int):
                for s1, trans in v.items():
                    if trans == "True":
                        program += "rel transition = {{({}, {})}}\n".format(s0, s1)
                    else:
                        program += "rel transition({},{}) = {}\n".format(s0, s1, sympy_to_scallop(trans))

        domains = {}
        for k, v in classes.items():
            if all([v2.isnumeric() for v2 in v.keys()]):
                domains[k] = [int(v2) for v2 in sorted(v.keys())]  # Map integers to themselves.
            else:
                domains[k] = sorted(v.values())  # Map categoricals to integer idx.

        output_mappings = {k: () for k in self.propositions}
        output_mappings.update({"next_state": task_opts["automaton"]["states"], "accepting": ()})

        input_mappings = {"prev_state": task_opts["automaton"]["states"]}
        input_mappings.update(domains)


        self.nesy_program = scallopy.Module(
            program=program,
            input_mappings=input_mappings,
            output_mappings=output_mappings,
            provenance=opts["constraint_module"]["provenance"],
            train_k=opts["constraint_module"]["train_k"],
            test_k=opts["constraint_module"]["test_k"],
            retain_graph=True,
            #early_discard=False,
        )


    def forward(self, log_s0, s0, imgs):
        img_labels = {v: self.backbone[v](imgs[i]) for i, v in enumerate(self.variables)}
        nesy_input = {k: F.softmax(v, dim=1) for k, v in img_labels.items()}
        nesy_input["prev_state"] = s0
        out = self.nesy_program(**nesy_input)

        unnormalized_next_state = out["next_state"]
        next_state = unnormalized_next_state / (torch.sum(unnormalized_next_state, dim=1, keepdim=True).detach() + 1e-7)

        return [img_labels[k] for k in sorted(img_labels.keys())], \
           [out[p] for p in sorted(out.keys()) if p.startswith("p_")], \
           torch.log(next_state + self.eps), next_state, out["accepting"]