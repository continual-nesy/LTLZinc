import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import pandas as pd
import os


class LTLZincIncrementalDataset:
    """
    Incremental dataset class. It encapsulates an entire curriculum of tasks.
    """
    def __init__(self, prefix_path, task_path, classes, split, use_random_samples, rng, transform=None):
        self.use_random_samples = use_random_samples
        self.rng = rng
        self.prefix_path = prefix_path
        self.df = pd.read_csv("{}/incremental_{}.csv".format(task_path, split)).fillna(value=-1)
        self.task_len = len(self.df["task_id"].unique())
        self.datasets = [LTLZincTaskDataset(self.df[self.df["task_id"] == i], prefix_path, classes, use_random_samples, rng, transform)
                         for i in range(self.task_len)]

    def __len__(self):
        return self.task_len

    def __getitem__(self, idx):
        return self.datasets[idx]


class LTLZincTaskDataset(Dataset):
    """
    Task dataset class. It contains samples belonging to a single task, as well as its available knowledge.
    """
    def __init__(self, df, prefix_path, classes, use_random_samples, rng, transform=None):
        super().__init__()

        self.df = df.reset_index()
        self.use_random_samples = use_random_samples
        self.rng = rng
        self.prefix_path = prefix_path

        self.vars = sorted(classes.keys())

        self.imgs = ["img_{}".format(v.split("_")[-1]) for v in self.vars]
        self.constraints = sorted([c for c in self.df.columns if c.startswith("p_")])

        self.classes = classes
        self.transform = transform

        if self.transform is not None:
            self.c, self.w, self.h = self.transform(read_image("{}/{}".format(self.prefix_path, self.df.loc[0][self.imgs[0]]))).shape
        else:
            self.c, self.w, self.h = read_image("{}/{}".format(self.prefix_path, self.df.loc[0][self.imgs[0]])).shape

        # State-level oracle. For each task, it returns the current state of the automaton equivalent to the LTLf formula.
        self.curr_state = int(self.df.loc[0]["transition"].split("->")[0])

        curr_mapping = {k: self.df.loc[0][k] for k in self.df.columns if k.startswith("p_")}

        # Predicate-level oracle. For each task, it returns the list of predicates true in that task.
        self.curr_preds = set()
        for k, v in curr_mapping.items():
            if "1" in str(v):
                # Assumption: we have a predicate-level oracle which annotates truth values ALSO for orphan predicates ("[p_0]", instead of "p_0").
                # This is a stronger knowledge injection than a state-level oracle, which only operates according to the temporal behavior.
                self.curr_preds.add(k)
        self.curr_preds = sorted(self.curr_preds)


        self.cached_paths = {}

        self.randomize_samples()

    def randomize_samples(self):
        self.randomized_imgs = []

        for idx in range(len(self)):
            seq = self.df[self.df["seq_id"] == idx].iloc[0]
            self.randomized_imgs.append({})
            for k in self.imgs:
                prefix = "/".join(seq[k].split("/")[:-1])
                if prefix not in self.cached_paths:
                    self.cached_paths[prefix] = os.listdir("{}/{}".format(self.prefix_path, prefix))
                new_file = self.rng.choice(self.cached_paths[prefix])
                img_path = "{}/{}".format(prefix, new_file)
                self.randomized_imgs[-1][k] = img_path


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx == 0 and self.use_random_samples:
            self.randomize_samples()

        seq = self.df[self.df["seq_id"] == idx].iloc[0]

        var_labels = torch.zeros(len(self.vars), dtype=torch.int64)
        imgs = torch.zeros(len(self.imgs), self.c, self.w, self.h)
        var_mask = torch.zeros(len(self.vars), dtype=torch.bool)

        for i, v in enumerate(self.vars):
            vv = v.replace("out_", "var_")
            # Convention: in the csv "class" refers to a value which affect the next state of the automaton.
            # "(class)" refers to a value which  DOES NOT affect the transition.
            if str(seq[vv]).startswith("("):
                var_labels[i] = self.classes[str(v)][seq[vv][1:-1]]
            else:
                var_labels[i] = self.classes[str(v)][str(seq[vv])]

                var_mask[i] = 1

            if self.use_random_samples:
                img_path = self.randomized_imgs[idx][self.imgs[i]]
            else:
                img_path = seq[self.imgs[i]]

            if self.transform is not None:
                imgs[i, :, :, :] = self.transform(read_image("{}/{}".format(self.prefix_path, img_path)))
            else:
                imgs[i, :, :, :] = read_image("{}/{}".format(self.prefix_path, img_path))

        return imgs, var_labels, var_mask
