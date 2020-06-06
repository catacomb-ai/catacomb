"""1) Include/define any dependencies for catacomb.System class"""
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
import catacomb

class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self):
        super(LightningMNISTClassifier, self).__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        x = x.view(batch_size, -1)
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)

        return x


"""2) Implementing catacomb.System class with initialization and output methods"""
class System(catacomb.System):
    def __init__(self):
        self.model = LightningMNISTClassifier.load_from_checkpoint('./saved_model.ckpt')

    # Implementing `output` interface for type `IMAGE -> LABEL`
    def output(self, image):
        # transform image and run through network
        pass
