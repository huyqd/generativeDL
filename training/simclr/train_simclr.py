from data import load_data
from trainer import train_simclr

if __name__ == "__main__":
    from lightning.pytorch.callbacks import ModelSummary  # noqa

    unlabeled_data, train_data_contrast = load_data()
    model = train_simclr(
        unlabeled_data=unlabeled_data,
        train_data_contrast=train_data_contrast,
        batch_size=256,
        hidden_dim=128,
        lr=5e-4,
        temperature=0.07,
        weight_decay=1e-4,
        max_epochs=500,
    )
