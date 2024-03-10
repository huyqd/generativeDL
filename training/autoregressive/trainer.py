import os

import lightning as L
import numpy as np
import torch
import torchvision
from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from lightning.pytorch.loggers import WandbLogger
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm

from training.autoregressive.models import MODEL_DICT

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


class ARModule(L.LightningModule):
    def __init__(self, model_name, c_in, c_hidden):
        super().__init__()
        self.save_hyperparameters()
        self.model = MODEL_DICT[model_name](c_in, c_hidden)
        self.example_input_array = torch.randn(1, 1, 28, 28)

    def forward(self, x):
        """Forward image through model and return logits for each pixel.

        Args:
            x: Image tensor with integer values between 0 and 255.
        """
        # Scale input from 0 to 255 back to -1 to 1
        return self.model(x)

    def likelihood(self, x):
        # Forward pass with bpd likelihood calculation
        pred = self.forward(x)
        nll = F.cross_entropy(pred, x, reduction="none")
        bpd = nll.mean(dim=[1, 2, 3]) * np.log2(np.exp(1))
        return bpd.mean()

    @torch.no_grad()
    def sample(self, img_shape, img=None):
        """Sampling function for the autoregressive model.

        Args:
            img_shape: Shape of the image to generate (B,C,H,W)
            img (optional): If given, this tensor will be used as
                             a starting image. The pixels to fill
                             should be -1 in the input tensor.
        """
        # Create empty image
        if img is None:
            img = torch.zeros(img_shape, dtype=torch.long).to(DEVICE) - 1
        # Generation loop
        for h in tqdm(range(img_shape[2]), leave=False, desc="Sampling images"):
            for w in range(img_shape[3]):
                for c in range(img_shape[1]):
                    # Skip if not to be filled (-1)
                    if (img[:, c, h, w] != -1).all().item():
                        continue
                    # For efficiency, we only have to input the upper part of the image
                    # as all other parts will be skipped by the masked convolutions anyway
                    pred = self(img[:, :, : h + 1, :])
                    probs = F.softmax(pred[:, :, c, h, w], dim=-1)
                    img[:, c, h, w] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
        return img

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss = self.likelihood(batch[0])
        self.log("train_bpd", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.likelihood(batch[0])
        self.log("val_bpd", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self.likelihood(batch[0])
        self.log("test_bpd", loss, prog_bar=True)


class GenerateCallback(Callback):
    def __init__(self, every_n_epochs=1):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    @staticmethod
    def _log_sampling_images(trainer, pl_module, n_images=16):
        samples = pl_module.sample(img_shape=(n_images, 1, 28, 28))
        nrow = min(n_images, 8)
        grid = torchvision.utils.make_grid(samples.cpu().float(), nrow=nrow, pad_value=128)
        wandb_logger.log_image(key="Sampling", images=[grid], step=trainer.global_step)

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._log_sampling_images(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            self._log_sampling_images(trainer, pl_module)


def train_autoregressive(
    model_name,
    train_loader,
    val_loader,
    test_loader,
    debug=False,
    **kwargs,
):
    callbacks = [
        ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd"),
        LearningRateMonitor("epoch"),
        GenerateCallback(every_n_epochs=1),
    ]
    logger = wandb_logger
    overfit_batches = 0
    if debug:
        callbacks += [ModelSummary(max_depth=-1)]
        logger = None
        overfit_batches = 10

    imgs = [train_loader.dataset[i][0].float() for i in range(32)]
    img_grid = torchvision.utils.make_grid(imgs, nrow=8, value_range=(0, 255))
    wandb_logger.log_image(key="Training data", images=[img_grid])

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "PixelCNN"),
        accelerator=ACCELERATOR,
        devices=1,
        max_epochs=50,
        callbacks=callbacks,
        logger=logger,
        overfit_batches=overfit_batches,
    )
    result = None
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "PixelCNN.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = ARModule.load_from_checkpoint(pretrained_filename)
        ckpt = torch.load(pretrained_filename, map_location=DEVICE)
        result = ckpt.get("result", None)
    else:
        model = ARModule(model_name, **kwargs)
        trainer.fit(model, train_loader, val_loader)
    model = model.to(DEVICE)

    if result is None:
        # Test best model on validation and test set
        val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
        test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
        result = {"test": test_result, "val": val_result}
    return model, result
