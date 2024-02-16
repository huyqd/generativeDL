import lightning as L
import torch
from metaflow import FlowSpec, step, batch

from data import load_data
from trainer import train_model
from utils import set_seed


class CifarFlow(FlowSpec):
    @batch(cpu=8, memory=32 * 1024, gpu=1)
    @step
    def start(self):
        set_seed(42)
        L.seed_everything(42)

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
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    CifarFlow()
