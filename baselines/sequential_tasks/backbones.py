import torch
import torch.nn.functional as F


class SmallNet(torch.nn.Module):  # Same as Scallop experiments (except with 3x32x32 inputs for CIFAR10 compatibility).
    def __init__(self, classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = torch.nn.Linear(64 * 5 * 5, 1024)
        self.fc2 = torch.nn.Linear(1024, classes)

        self.last_layer = self.fc2

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x


class NonZeroMNIST(torch.nn.Module):  # Used for parameter sharing while preventing division by zero in Scallop.
    def __init__(self, mnistnet):
        super().__init__()
        self.mnistnet = mnistnet

        self.last_layer = self.mnistnet.fc2

    def forward(self, x):
        return self.mnistnet(x)[:, 1:]

backbone_factory = {}

class FiveFMNIST(torch.nn.Module):  # Used for parameter sharing while preventing out of domain values.
    def __init__(self, mnistnet):
        super().__init__()
        self.mnistnet = mnistnet

        self.last_layer = self.mnistnet.fc2

    def forward(self, x):
        return self.mnistnet(x)[:, 5:] # Simply return the *last* five neurons.

backbone_factory = {}

backbone_factory["independent"] = { # Every input stream is processed by a different neural network.
    "sequential/task1": lambda: {"var_u": SmallNet(5), "var_v": SmallNet(5), "var_w": SmallNet(5), "var_x": SmallNet(5), "var_y": SmallNet(10), "var_z": SmallNet(10)},
    "sequential/task2": lambda: {"var_t": SmallNet(5), "var_u": SmallNet(5), "var_v": SmallNet(5), "var_w": SmallNet(5), "var_x": SmallNet(5), "var_y": SmallNet(10), "var_z": SmallNet(10)},
    "sequential/task3": lambda: {"var_x": SmallNet(10), "var_y": SmallNet(10), "var_z": SmallNet(10)},
    "sequential/task4": lambda: {"var_x": SmallNet(10), "var_y": SmallNet(10), "var_z": SmallNet(10)},
    "sequential/task5": lambda: {"var_w": SmallNet(10), "var_x": SmallNet(10), "var_y": SmallNet(10), "var_z": SmallNet(10)},
    "sequential/task6": lambda: {"var_x": SmallNet(10), "var_y": SmallNet(10), "out_z": SmallNet(10)},
}

backbone_factory["dataset"] = { # Input streams processing the same dataset, with the same values, are processed by the same neural network.
    "sequential/task1": lambda: {"var_y": (fmnist := SmallNet(10)), "var_z": fmnist, "var_v": (small_fmnist := FiveFMNIST(fmnist)), "var_w": small_fmnist, "var_x": small_fmnist},
    "sequential/task2": lambda: {"var_y": (fmnist := SmallNet(10)), "var_z": fmnist, "var_v": (small_fmnist := FiveFMNIST(fmnist)), "var_w": small_fmnist, "var_x": small_fmnist},
    "sequential/task3": lambda: {"var_x": (net := SmallNet(10)), "var_y": net, "var_z": net},
    "sequential/task4": lambda: {"var_x": SmallNet(10), "var_y": (net := SmallNet(10)), "var_z": net},
    "sequential/task5": lambda: {"var_w": (net := SmallNet(10)), "var_x": net, "var_y": net, "var_z": net},
    "sequential/task6": lambda: {"var_x": (net := SmallNet(10)), "var_y": net, "out_z": net},
}

# For domain adaptation tasks, it may be useful to define a ["domain"] entry, where input streams processing different datasets, but with the same labels are processed by the same neural network.
# e.g. for task4 it would be {"var_x": (net := SmallNet(10)), "var_y": net, "var_z": net} (var_x comes from MNIST, while var_y and var_z come from FMNIST, but labels are mapped to the same values).

# Duplicating entries for the long-sequence variants of each task.
for v in backbone_factory.values():
    v.update({k2.replace("/", "_long/"): v2 for k2, v2 in v.items()})