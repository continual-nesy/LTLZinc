import torch
import torch.nn.functional as F


class SmallNet(torch.nn.Module):  # Same as Scallop experiments (except with 3x32x32 inputs for CIFAR10 compatibility).
    def __init__(self, classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = torch.nn.Linear(64 * 5 * 5, 1024)
        self.fc2 = torch.nn.Linear(1024, classes)

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

    def forward(self, x):
        return self.mnistnet(x)[:, 1:]

backbone_factory = {}

class FiveFMNIST(torch.nn.Module):  # Used for parameter sharing while preventing out of domain values.
    def __init__(self, mnistnet):
        super().__init__()
        self.mnistnet = mnistnet

    def forward(self, x):
        return self.mnistnet(x)[:, :5]

backbone_factory = {}

backbone_factory["independent"] = { # Every input stream is processed by a different neural network.
    "task1": lambda: {"out_t": SmallNet(5), "out_u": SmallNet(5), "out_v": SmallNet(5), "out_w": SmallNet(5), "out_x": SmallNet(5), "var_y": SmallNet(10), "var_z": SmallNet(10)},
    "task2": lambda: {"var_x": SmallNet(10), "var_y": SmallNet(9), "out_z": SmallNet(10)},
    "task3": {"var_u": SmallNet(10), "var_v": SmallNet(10), "var_w": SmallNet(10), "out_x": SmallNet(10), "out_y": SmallNet(10), "out_z": SmallNet(10)},
    "task4": lambda: {"var_x": SmallNet(10), "out_y": SmallNet(10), "var_z": SmallNet(10)},
    "task5": lambda: {"var_x": SmallNet(10), "out_y": SmallNet(10), "var_z": SmallNet(10)},
    "task6": lambda: {"var_x": SmallNet(10), "var_y": SmallNet(10), "out_z": SmallNet(10)},
    "task7": lambda: {"out_t": SmallNet(5), "out_u": SmallNet(5), "out_v": SmallNet(5), "out_w": SmallNet(5), "out_x": SmallNet(5), "var_y": SmallNet(10), "var_z": SmallNet(10)},
    "task8": lambda: {"var_x": SmallNet(10), "var_y": SmallNet(10), "out_z": SmallNet(10)},
}

backbone_factory["dataset"] = { # Input streams processing the same dataset, with the same values, are processed by the same neural network.
    "task1": lambda: {"var_y": (fmnist := SmallNet(10)), "var_z": fmnist, "out_t": (small_fmnist := FiveFMNIST(fmnist)), "out_u": small_fmnist, "out_v": small_fmnist, "out_w": small_fmnist, "out_x": small_fmnist},
    "task2": lambda: {"var_x": (net := SmallNet(10)), "var_y": NonZeroMNIST(net), "out_z": net}, # Exclude var_y = 0 from domain, but still share the same network.
    "task3": lambda: {"var_u": (net := SmallNet(10)), "var_v": net, "var_w": net, "out_x": net, "out_y": net, "out_z": net},
    "task4": lambda: {"var_x": (net := SmallNet(10)), "out_y": net, "var_z": net},
    "task5": lambda: {"var_x": (net := SmallNet(10)), "out_y": net, "var_z": net},
    "task6": lambda: {"var_x": (net := SmallNet(10)), "var_y": net, "out_z": net},
    "task7": lambda: {"out_t": (cifar := SmallNet(5)), "out_u": cifar, "out_v": cifar, "out_w": cifar, "out_x": cifar, "var_y": (fmnist := SmallNet(10)), "var_z": fmnist}, # Same as task 1 but cross-domain (disjoint semantics)
    "task8": lambda: {"var_x": (net := SmallNet(10)), "var_y": net, "out_z": SmallNet(10)} # Same as task6, but cross-domain (shared semantics): MNIST 0-9 (x,y) and SVHN 0-9 (z) use different nets.
}

backbone_factory["domain"] = { # Input streams sharing domain values are processed by the same neural network.
    "task1": backbone_factory["dataset"]["task1"], # Since each domain corresponds to a dataset, copy ["dataset"]
    "task2": backbone_factory["dataset"]["task2"],
    "task3": backbone_factory["dataset"]["task3"],
    "task4": backbone_factory["dataset"]["task4"],
    "task5": backbone_factory["dataset"]["task5"],
    "task6": backbone_factory["dataset"]["task6"],
    "task7": backbone_factory["dataset"]["task6"],
    "task8": lambda: {"var_x": (net := SmallNet(10)), "var_y": net, "out_z": net} # MNIST 0-9 (x,y) and SVHN 0-9 (z) use the same net.
}

# Duplicating entries for the long-sequence variants of each task.
for v in backbone_factory.values():
    v.update({"{}_long".format(k2): v2 for k2, v2 in v.items()})
