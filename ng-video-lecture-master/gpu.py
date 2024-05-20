import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from metaflow import FlowSpec, step, Parameter, batch, JSONType


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flatten all dims except batch
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    return trainloader, testloader, classes


def train_model(trainloader, lr, epochs=1):
    net = Net()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(f"Training on {device}")

    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            inputs = data[0].to(device)
            labels = data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return net


def run_inference_and_tests(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct // total


class GPUFlow(FlowSpec):
    learning_rates = Parameter("learning-rates", default=json.dumps([0.01, 0.001]), type=JSONType)

    @step
    def start(self):
        data = load_data()
        self.trainloader = data[0]
        self.testloader = data[1]
        self.classes = data[2]
        self.next(self.train, foreach="learning_rates")

    @batch(
        gpu=1,
        cpu=8,
        memory=32 * 1024,
        image="475126315063.dkr.ecr.eu-central-1.amazonaws.com/ltr-hotel/metaflow:gpu",
        queue="arm-metaflow-memory",
    )
    @step
    def train(self):
        self.model = train_model(self.trainloader, lr=self.input)
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        result = run_inference_and_tests(self.model, self.testloader)
        self.accuracy = result
        self.next(self.join)

    @step
    def join(self, inputs):
        best_model = None
        best_score = -1
        for i in inputs:
            if i.accuracy > best_score:
                best_score = i.accuracy
                best_model = i.model
        self.best_model = best_model
        self.best_score = best_score
        print(f"Best model accuracy was {best_score}%.")
        self.next(self.end)

    @step
    def end(self):
        print("Done")


if __name__ == "__main__":
    GPUFlow()
