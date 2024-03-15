from data import load_data
from trainer import train_autoregressive

if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data()
    model, result = train_autoregressive(
        "MyPixelCNN",
        train_loader,
        val_loader,
        test_loader,
        c_in=1,
        c_hidden=64,
    )

    test_res = result["test"][0]
    print(
        f"Test bits per dimension: {test_res['test_loss'] if 'test_loss' in test_res else test_res['test_bpd']:4.3f}bpd"
    )
