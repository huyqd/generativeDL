import matplotlib
import pytorch_lightning as pl
import seaborn as sns
import torch

from data import load_data
from trainer import train_model
from utils import set_seed

matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()

if __name__ == "__main__":
    set_seed(42)
    pl.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, val_loader, test_loader = load_data()
    googlenet_model, googlenet_results = train_model(
        model_name="GoogleNet",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_hparams={"num_classes": 10, "act_fn_name": "relu"},
        optimizer_name="Adam",
        optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
    )
