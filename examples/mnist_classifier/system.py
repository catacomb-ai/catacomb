"""1) Include/define any dependencies for catacomb.System class"""
import base64
from io import BytesIO
import pytorch_lightning as pl
import torch
import catacomb

from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


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
        checkpoint = torch.load('./saved_model.ckpt', map_location=lambda storage, loc: storage)
        self.model = LightningMNISTClassifier()
        self.model.load_state_dict(checkpoint['state_dict'])

    # Implementing `output` interface for type `IMAGE -> LABEL`
    def output(self, file):
        # decode base64 image and open in PIL
        header, encoded = file.split(',', 1)
        decoded = BytesIO(base64.b64decode(encoded))
        img = Image.open(decoded)

        # transform PIL image to tensor
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])        
        img = transform(img).unsqueeze(0)
        
        # compute prediction
        prediction = self.model(img).argmax(1).item()
        return prediction

if __name__ == '__main__':
    catacomb.start(System())