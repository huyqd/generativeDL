import lightning as L
import torch
from torch.utils import data as data
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR_DATASET_PATH = "../../assets/cifar"


def load_cifar_data():
    # Transformations applied on each image => only make them a tensor
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = CIFAR10(root=CIFAR_DATASET_PATH, train=True, transform=transform, download=True)
    L.seed_everything(42)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])

    # Loading the test set
    test_set = CIFAR10(root=CIFAR_DATASET_PATH, train=False, transform=transform, download=True)

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
        batch_size=256,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        persistent_workers=True,
    )
    test_loader = data.DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader
