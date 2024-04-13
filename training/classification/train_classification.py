import lightning as L
import torch

from trainer import train_model
from data import load_cifar_data
from utils import set_seed

if __name__ == "__main__":
    from lightning.pytorch.callbacks import ModelSummary  # noqa

    set_seed(42)
    L.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, val_loader, test_loader = load_cifar_data()
    # model, results = train_model(
    #     model_name="GoogleNet",
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     test_loader=test_loader,
    #     model_hparams={"num_classes": 10, "act_fn_name": "relu"},
    #     optimizer_name="Adam",
    #     optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
    #     trainer_args={
    #         "overfit_batches": 10,
    #         "callbacks": [
    #             ModelSummary(max_depth=-1),
    #         ],
    #     },
    # )

    # model, results = train_model(
    #     model_name="ResNet",
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     test_loader=test_loader,
    #     model_hparams={
    #         "num_classes": 10,
    #         "c_hidden": [16, 32, 64],
    #         "num_blocks": [3, 3, 3],
    #         "act_fn_name": "relu",
    #         "resnet_block_name": "PreActResNetBlock",
    #     },
    #     optimizer_name="SGD",
    #     optimizer_hparams={"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
    #     trainer_args={
    #         "overfit_batches": 10,
    #         "callbacks": [
    #             ModelSummary(max_depth=-1),
    #         ],
    #     },
    # )

    # model, results = train_model(
    #     model_name="DenseNet",
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     test_loader=test_loader,
    #     model_hparams={
    #         "num_classes": 10,
    #         "num_layers": [6, 6, 6, 6],
    #         "bn_size": 2,
    #         "growth_rate": 16,
    #         "act_fn_name": "relu",
    #     },
    #     optimizer_name="Adam",
    #     optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
    #     trainer_args={
    #         "overfit_batches": 10,
    #         "callbacks": [
    #             ModelSummary(max_depth=-1),
    #         ],
    #     },
    # )

    model, results = train_model(
        model_name="VisionTransformer",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_hparams={
            "embed_dim": 256,
            "hidden_dim": 512,
            "num_heads": 8,
            "num_layers": 6,
            "patch_size": 4,
            "num_channels": 3,
            "num_patches": 64,
            "num_classes": 10,
            "dropout": 0.2,
        },
        optimizer_name="Adam",
        optimizer_hparams={"lr": 3e-4, "weight_decay": 1e-4},
        trainer_args={
            # "overfit_batches": 10,
            "callbacks": [
                ModelSummary(max_depth=-1),
            ],
        },
    )
