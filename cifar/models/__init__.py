from models.inception import SolutionGoogleNet
from models.convnet import SimpleConvNet

MODEL_DICT = {
    "GoogleNet": SolutionGoogleNet,
    "SimpleConvNet": SimpleConvNet,
}
