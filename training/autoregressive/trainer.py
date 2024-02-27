import os

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from training.autoregressive.models.pixelcnn import PixelCNN

wandb_logger = WandbLogger(log_model=False, project="autoregressive", save_dir="../../assets/logs/")

CHECKPOINT_PATH = "../../assets/saved_models/"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    ACCELERATOR = "gpu"
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    ACCELERATOR = "mps"
else:
    DEVICE = torch.device("cpu")
    ACCELERATOR = "cpu"


def train_autoregressive(train_loader, val_loader, test_loader, **kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "PixelCNN"),
        accelerator=ACCELERATOR,
        devices=1,
        max_epochs=150,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd"),
            LearningRateMonitor("epoch"),
        ],
    )
    result = None
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "PixelCNN.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = PixelCNN.load_from_checkpoint(pretrained_filename)
        ckpt = torch.load(pretrained_filename, map_location=DEVICE)
        result = ckpt.get("result", None)
    else:
        model = PixelCNN(**kwargs)
        trainer.fit(model, train_loader, val_loader)
    model = model.to(DEVICE)

    if result is None:
        # Test best model on validation and test set
        val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
        test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
        result = {"test": test_result, "val": val_result}
    return model, result
