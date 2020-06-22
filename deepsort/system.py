"""1) Include/define any dependencies for catacomb.System class"""
import torch
import catacomb

from data import str_to_array, tensor_from_list
from evaluate import evaluate
from models.encoder import Encoder
from models.ptr_decoder import PtrDecoder


"""2) Implementing catacomb.System class with initialization and output methods"""
class System(catacomb.System):
    def __init__(self):
        hidden_dim = embedding_dim = 128
        data_dim =  101

        self.encoder = Encoder(input_dim=data_dim,
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim)

        self.decoder = PtrDecoder(output_dim=data_dim,
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim)

        checkpoint = torch.load('./e1i0.ckpt', map_location='cpu')

        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])

    def output(self, unsorted):
        unsorted_list = str_to_array(unsorted['list'])
        unsorted_tensor = tensor_from_list(unsorted_list)

        return str(evaluate(self.encoder, self.decoder, unsorted_tensor, True))
