from torchvision import transforms
from torchvision.datasets import STL10

DATASET_PATH = "../../assets/stl10"


class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


def load_data():
    # Transformations applied on each image => only make them a tensor
    contrast_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=96),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    unlabeled_data = STL10(
        root=DATASET_PATH,
        split="unlabeled",
        download=True,
        transform=ContrastiveTransformations(contrast_transforms, n_views=2),
    )
    train_data_contrast = STL10(
        root=DATASET_PATH,
        split="train",
        download=True,
        transform=ContrastiveTransformations(contrast_transforms, n_views=2),
    )

    return unlabeled_data, train_data_contrast
