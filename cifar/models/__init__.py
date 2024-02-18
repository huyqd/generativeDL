from models.inception import GoogleNet
from models.convnet import SimpleConvNet
from models.resnet import ResNet

MODEL_DICT = {
    "GoogleNet": GoogleNet,
    "SimpleConvNet": SimpleConvNet,
    "ResNet": ResNet,
}
