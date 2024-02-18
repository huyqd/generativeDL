import random

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def display_images(train_dataset, test_transform):
    NUM_IMAGES = 4
    images = [train_dataset[idx][0] for idx in range(NUM_IMAGES)]
    orig_images = [Image.fromarray(train_dataset.data[idx]) for idx in range(NUM_IMAGES)]
    orig_images = [test_transform(img) for img in orig_images]

    img_grid = torchvision.utils.make_grid(
        torch.stack(images + orig_images, dim=0), nrow=4, normalize=True, pad_value=0.5
    )
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(8, 8))
    plt.title("Augmentation examples on CIFAR10")
    plt.imshow(img_grid)
    plt.axis("off")
    plt.show()
    plt.close()
