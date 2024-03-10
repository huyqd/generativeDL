import lightning as L
from metaflow import FlowSpec, step, batch

from data import load_data
from trainer import train_autoregressive


class TrainFlow(FlowSpec):
    @batch(cpu=8, memory=30 * 1024, gpu=1)
    @step
    def start(self):
        L.seed_everything(42)

        train_loader, val_loader, test_loader = load_data()
        model, result = train_autoregressive(
            "GatedPixelCNN",
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
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    TrainFlow()
