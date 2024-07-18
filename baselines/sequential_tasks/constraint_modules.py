import torch
import torch.nn.functional as F
import scallopy

class ConstraintMLP(torch.nn.Module):
    def __init__(self, num_neurons, num_classes, propositions):
        super().__init__()

        self.propositions = propositions

        self.net = torch.nn.ModuleDict(
            {p: torch.nn.Sequential(
                torch.nn.Linear(num_classes, num_neurons),
                torch.nn.ReLU(),
                torch.nn.Linear(num_neurons, 1),
                torch.nn.Sigmoid()
            ) for p in propositions})

    def forward(self, imgs):
        input_tensor = torch.cat([imgs[i] for i in sorted(imgs.keys())], dim=1)

        return {p: self.net[p](input_tensor).squeeze(1) for p in self.net.keys()}

class ConstraintScallop(torch.nn.Module):
    def __init__(self, program_path, classes, propositions, provenance, train_k, test_k):
        super().__init__()

        domains = {}
        for k, v in classes.items():
            if all([v2.isnumeric() for v2 in v.keys()]):
                domains[k] = [int(v2) for v2 in sorted(v.keys())]  # Map integers to themselves.
            else:
                domains[k] = sorted(v.values())  # Map categoricals to integer idx.

        self.propositions = propositions

        self.net = scallopy.Module(
            file="{}/scallop_constraints.scl".format(program_path),
            input_mappings=domains,
            output_mappings={k: () for k in self.propositions},
            provenance=provenance,
            train_k=train_k,
            test_k=test_k,
            retain_graph=True,
            #early_discard=False,
            )

    def forward(self, imgs):
        nesy_in = {k: F.softmax(v, dim=1) for k,v in imgs.items()}
        nesy_out = self.net(**nesy_in)

        # If there is only one output, Scallop returns a tensor, instead of a dict of tensors. Fixing it.
        if not isinstance(nesy_out, dict):
            nesy_out = {self.propositions[0]: nesy_out}

        return nesy_out
