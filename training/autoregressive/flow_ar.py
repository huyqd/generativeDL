from metaflow import FlowSpec, step, batch

from train_ar import train


class TrainFlow(FlowSpec):
    @batch(cpu=8, memory=32 * 1024, gpu=1)
    @step
    def start(self):
        train()
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    TrainFlow()
