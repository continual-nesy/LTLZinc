import os
import tqdm
from torchvision import transforms, datasets

# Simple downloader utility. It downloads torchvision datasets, transforms images into a unique shape (32x32 RGB), and sorts them into subfolders of data/.
# These datasets are intended for the provided benchmarks; for your own datasets, simply organize them into folders: data/{dataset name}/{split}/{class id}.

def fmnist_labeler(x):
    labels = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "boot"]
    return labels[x]

def emnist_labeler(x):
    return "{:c}".format(97 + x)

def cifar_labeler(x):
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    return labels[x]

if __name__ == "__main__":
    ds = {
        "train": {
            "mnist": datasets.MNIST("tmp", train=True, transform=transforms.CenterCrop(32), download=True),
            "fmnist": datasets.FashionMNIST("tmp", train=True, transform=transforms.CenterCrop(32), target_transform=fmnist_labeler, download=True),
            "kmnist": datasets.KMNIST("tmp", train=True, transform=transforms.CenterCrop(32), download=True),
            "cifar10": datasets.CIFAR10("tmp", train=True, target_transform=cifar_labeler, download=True),
            "svhn": datasets.SVHN("tmp", split="train", download=True),
        },
        "test": {
            "mnist": datasets.MNIST("tmp", train=False, transform=transforms.CenterCrop(32), download=True),
            "fmnist": datasets.FashionMNIST("tmp", train=False, transform=transforms.CenterCrop(32), target_transform=fmnist_labeler, download=True),
            "kmnist": datasets.KMNIST("tmp", train=False, transform=transforms.CenterCrop(32), download=True),
            "cifar10": datasets.CIFAR10("tmp", train=False, target_transform=cifar_labeler, download=True),
            "svhn": datasets.SVHN("tmp", split="test", download=True),
        }
    }

    for split, v in tqdm.tqdm(ds.items(), position=0, desc="split", leave=False):
        for name, vv in tqdm.tqdm(v.items(), position=1, desc="dataset", leave=False):
            path = "data/{}/{}".format(split, name)
            for i, x in tqdm.tqdm(enumerate(vv), position=2, desc="sample"):
                os.makedirs("{}/{}".format(path, x[1]), exist_ok=True)
                x[0].convert("RGB").save("{}/{}/{}.png".format(path, x[1], i), "PNG")