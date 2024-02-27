import lightning as L

from data import load_data
from trainer import train_autoregressive
from training.autoregressive.utils import show_imgs

if __name__ == "__main__":
    from lightning.pytorch.callbacks import ModelSummary  # noqa

    train_loader, val_loader, test_loader = load_data()
    model, result = train_autoregressive(train_loader, val_loader, test_loader, c_in=1, c_hidden=64)
    test_res = result["test"][0]
    print(
        "Test bits per dimension: %4.3fbpd"
        % (test_res["test_loss"] if "test_loss" in test_res else test_res["test_bpd"])
    )

    L.seed_everything(1)
    samples = model.sample(img_shape=(16, 1, 28, 28))
    show_imgs(samples.cpu())