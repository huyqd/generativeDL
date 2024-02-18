from trainer import train_model
from data import load_data

if __name__ == "__main__":
    from lightning.pytorch.callbacks import ModelSummary  # noqa

    train_loader, test_loader = load_data()
    model = train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        img_shape=(1, 28, 28),
        batch_size=train_loader.batch_size,
        lr=1e-4,
        beta1=0.0,
    )
