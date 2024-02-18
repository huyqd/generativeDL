from models.inception import GoogleNet
from models.convnet import SimpleConvNet

MODEL_DICT = {
    "GoogleNet": GoogleNet,
    "SimpleConvNet": SimpleConvNet,
}
