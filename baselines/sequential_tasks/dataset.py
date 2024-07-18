import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import pandas as pd

class LTLZincSequenceDataset(Dataset):
    def __init__(self, prefix_path, task_path, classes, split, transform=None):
        super().__init__()

        self.prefix_path = prefix_path
        self.df = pd.read_csv("{}/sequence_{}.csv".format(task_path, split)).fillna(value=-1)

        self.vars = sorted(classes.keys())

        self.imgs = ["img_{}".format(v.split("_")[-1]) for v in self.vars]
        self.constraints = sorted([c for c in self.df.columns if c.startswith("p_")])

        self.seq_len = self.df.groupby("seq_id")["seq_id"].count().max()


        self.classes = classes

        self.c, self.w, self.h = read_image("{}/{}".format(self.prefix_path, self.df.loc[0][self.imgs[0]])).shape

        self.transform = transform

    def __len__(self):
        return len(self.df["seq_id"].unique())

    def __getitem__(self, idx):
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

                if self.transform is not None:
                    imgs[j][i, :, :, :] = self.transform(read_image("{}/{}".format(self.prefix_path, seq[self.imgs[j]])))
                else:
                    imgs[j][i, :, :, :] = read_image("{}/{}".format(self.prefix_path, seq[self.imgs[j]]))

        return imgs, states, var_labels, cons_labels, seq_label, seq_mask, var_mask, cons_mask