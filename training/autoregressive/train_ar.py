import click

from data import load_deepul_data
from trainer import train_autoregressive

train_config = {
    "batch_size": 128,
    "num_workers": 4,
    "persistent_workers": True,
    "pin_memory": True,
    "epochs": 10,
    "data_name": "mnist",
    "model_name": "PixelCNN",
    "model_params": {
        # "use_resblock": True,
        # "n_filters": 120,
        # "n_layers": 8,
    },
    "sampling_every_n_epochs": 5,
}


@click.command()
@click.option("--debug", default=False)
@click.option("--data-name", default="shapes")
def train(debug, data_name):
    train_config["debug"] = debug
    train_config["data_name"] = data_name
    train_loader, val_loader, test_loader = load_deepul_data(train_config)
    model, result = train_autoregressive(train_loader, val_loader, test_loader, train_config)

    return model, result


if __name__ == "__main__":
    model, result = train()
