import pickle
from pathlib import Path

import lightning as L
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST

current_path = Path(__file__).parent
assets_path = current_path.parent.parent / "assets"
mnist_path = assets_path / "mnist"
deepul_path = assets_path / "deepul"


def load_pickled_data(fname: str, include_labels: bool = False):
    with open(fname, "rb") as f:
        data = pickle.load(f)

    train_data, test_data = data["train"], data["test"]
    if "mnist.pkl" in fname or "shapes.pkl" in fname:
        # Binarize MNIST and shapes dataset
        train_data = (train_data > 127.5).astype("uint8")
        test_data = (test_data > 127.5).astype("uint8")
    if "celeb.pkl" in fname:
        train_data = train_data[:, :, :, [2, 1, 0]]
        test_data = test_data[:, :, :, [2, 1, 0]]
    if include_labels:
        return train_data, test_data, data["train_labels"], data["test_labels"]
    return train_data, test_data


def load_deepul_data(train_config):
    train_data, test_data = load_pickled_data(str(deepul_path / f"{train_config['data_name']}.pkl"))
    train_data = torch.from_numpy(train_data.transpose(0, 3, 1, 2))
    test_data = torch.from_numpy(test_data.transpose(0, 3, 1, 2))
    train_set, test_set = data.TensorDataset(train_data), data.TensorDataset(test_data)
    input_shape = tuple(train_data.shape[1:])
    train_config["model_params"].update({"input_shape": input_shape, "n_colors": input_shape[0] + 1})

    train_loader = data.DataLoader(
        train_set,
        batch_size=train_config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=train_config["num_workers"],
        persistent_workers=train_config["persistent_workers"],
        pin_memory=train_config["pin_memory"],
    )
    val_loader = data.DataLoader(
        test_set,
        batch_size=train_config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=train_config["num_workers"],
        persistent_workers=train_config["persistent_workers"],
        pin_memory=train_config["pin_memory"],
    )

    return train_loader, val_loader, None


# Convert images from 0-1 to 0-255 (integers). We use the long datatype as we will use the images as labels as well
def discretize(sample):
    return (sample * 255).to(torch.long)


def load_mnist_data():
    # Transformations applied on each image => only make them a tensor
    transform = transforms.Compose([transforms.ToTensor(), discretize])

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = MNIST(root=mnist_path, train=True, transform=transform, download=True)
    L.seed_everything(42)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

    # Loading the test set
    test_set = MNIST(root=mnist_path, train=False, transform=transform, download=True)

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(
        train_set,
        batch_size=256,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        persistent_workers=True,
    )
    test_loader = data.DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train, _, _ = load_deepul_data(
        {
            "data_name": "shapes",
            "batch_size": 128,
            "num_workers": 4,
            "persistent_workers": True,
            "pin_memory": True,
        }
    )
