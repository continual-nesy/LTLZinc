import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import pandas as pd
import os

class LTLZincSequenceDataset(Dataset):
    def __init__(self, prefix_path, task_path, classes, split, use_random_samples, rng, transform=None):
        super().__init__()

        self.use_random_samples = use_random_samples
        self.rng = rng
        self.prefix_path = prefix_path
        self.df = pd.read_csv("{}/sequence_{}.csv".format(task_path, split)).fillna(value=-1)

        self.vars = sorted(classes.keys())

        self.imgs = ["img_{}".format(v.split("_")[-1]) for v in self.vars]
        self.constraints = sorted([c for c in self.df.columns if c.startswith("p_")])

        self.seq_len = self.df.groupby("seq_id")["seq_id"].count().max()


        self.classes = classes

        self.c, self.w, self.h = read_image("{}/{}".format(self.prefix_path, self.df.loc[0][self.imgs[0]])).shape

        self.transform = transform
        self.cached_paths = {}
        self.randomize_samples()

    def randomize_samples(self):
        self.randomized_imgs = []


        for idx in range(len(self)):
            seqs = self.df[self.df["seq_id"] == idx].reset_index()
            self.randomized_imgs.append([])
            for i, seq in seqs.iterrows():
                self.randomized_imgs[-1].append({})
                for k in self.imgs:
                    prefix = "/".join(seq[k].split("/")[:-1])
                    if prefix not in self.cached_paths:
                        self.cached_paths[prefix] = os.listdir("{}/{}".format(self.prefix_path, prefix))
                    new_file = self.rng.choice(self.cached_paths[prefix])
                    img_path = "{}/{}".format(prefix, new_file)
                    self.randomized_imgs[-1][-1][k] = img_path

    def __len__(self):
        return len(self.df["seq_id"].unique())

    def __getitem__(self, idx):
        if idx == 0 and self.use_random_samples:
            self.randomize_samples()

        seqs = self.df[self.df["seq_id"] == idx].reset_index()
        var_labels = [torch.zeros(self.seq_len, dtype=torch.int64) for _ in self.vars] # List of tensors (num_var, seq_len, classes).
        cons_labels = [torch.zeros(self.seq_len) for _ in self.constraints]

        imgs = [torch.zeros(self.seq_len, self.c, self.w, self.h) for _ in self.imgs]
        states = -torch.ones((self.seq_len + 1), dtype=torch.int64)

        seq_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        var_mask = [torch.zeros(self.seq_len, dtype=torch.bool) for v in self.vars]
        cons_mask = [torch.zeros(self.seq_len, dtype=torch.bool) for _ in self.constraints]

        states[0] = 0
        for i, seq in seqs.iterrows():
            seq_mask[i] = 1.

            seq_label = torch.tensor(seq["accepted"], dtype=torch.float)
            s0, s1 = seq["transition"].split("->")
            states[i + 1] = int(s1)

            for j, p in enumerate(self.constraints):
                if str(seq[p]).startswith("("):
                    cons_labels[j][i] = int(seq[p][1:-1])
                else:
                    cons_labels[j][i] = int(seq[p])
                    cons_mask[j][i] = 1

            for j, v in enumerate(self.vars):
                vv = v.replace("out_", "var_")
                # Convention: in the csv "0" and "1" refer to false/true which affect the next state of the automaton.
                # "(0)" and "(1)" refer to "false but don't care" and "true but don't care",
                # because they are irrelevant in the current state.
                if str(seq[vv]).startswith("("):
                    var_labels[j][i] = self.classes[str(v)][seq[vv][1:-1]]
                else:
                    var_labels[j][i] = self.classes[str(v)][str(seq[vv])]

                    var_mask[j][i] = 1

                if self.use_random_samples:
                    img_path = self.randomized_imgs[idx][i][self.imgs[j]]
                else:
                    img_path = seq[self.imgs[j]]

                if self.transform is not None:
                    imgs[j][i, :, :, :] = self.transform(read_image("{}/{}".format(self.prefix_path, img_path)))
                else:
                    imgs[j][i, :, :, :] = read_image("{}/{}".format(self.prefix_path, img_path))

        return imgs, states, var_labels, cons_labels, seq_label, seq_mask, var_mask, cons_mask
