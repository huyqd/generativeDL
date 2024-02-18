from torch.utils import data as data
from torchvision import transforms
from torchvision.datasets import MNIST

DATASET_PATH = "../data/mnist"


def load_data():
    # Transformations applied on each image => make them a tensor and normalize between -1 and 1
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Loading the training dataset. We need to split it into a training and validation part
    train_set = MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)

    # Loading the test set
    test_set = MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

    # We define a set of data loaders that we can use for various purposes later.
    # Note that for actually training a model, we will use different data loaders
    # with a lower batch size.
    train_loader = data.DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
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

    return train_loader, test_loader
