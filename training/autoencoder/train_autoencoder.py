from data import load_cifar_data
from trainer import train_cifar

if __name__ == "__main__":
    from lightning.pytorch.callbacks import ModelSummary  # noqa

    train_loader, val_loader, test_loader = load_cifar_data()
    model = train_cifar(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        latent_dim=128,
    )
