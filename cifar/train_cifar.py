import lightning as L
import torch

from cifar.trainer import train_model
from data import load_data
from utils import set_seed

if __name__ == "__main__":
    from lightning.pytorch.callbacks import ModelSummary

    set_seed(42)
    L.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, val_loader, test_loader = load_data()
    model, results = train_model(
        model_name="GoogLeNet",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_hparams={"num_classes": 10, "act_fn_name": "relu"},
        optimizer_name="Adam",
        optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
        trainer_args={
            "overfit_batches": 10,
            "callbacks": [
                ModelSummary(max_depth=-1),
            ],
        },
    )
