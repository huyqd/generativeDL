import lightning as L
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST

DATASET_PATH = "../../assets/mnist"


# Convert images from 0-1 to 0-255 (integers)
def discretize(sample):
    return (sample * 255).to(torch.int32)


def load_data():
    # Transformations applied on each image => make them a tensor and discretize
    transform = transforms.Compose([transforms.ToTensor(), discretize])

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
    L.seed_everything(42)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

    # Loading the test set
    test_set = MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

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
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        persistent_workers=True,
    )
    test_loader = data.DataLoader(
        test_set,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader
