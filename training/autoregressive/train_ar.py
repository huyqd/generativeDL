from data import load_deepul_data
from trainer import train_autoregressive

train_config = {
    "batch_size": 128,
    "num_workers": 4,
    "persistent_workers": True,
    "pin_memory": True,
    "data_name": "shapes",
    "model_name": "PixelCNN",
    "model_params": {},
    "debug": False,
    "epochs": 10,
}

if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_deepul_data(train_config)
    model, result = train_autoregressive(train_loader, val_loader, test_loader, train_config)
