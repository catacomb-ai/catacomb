"""1) Include/define any dependencies for catacomb.System class"""
import torch
import torchtext
import torch.nn as nn
import en_core_web_sm
import catacomb
from catacomb import Types

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):        
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden.squeeze(0))


"""2) Implementing catacomb.System class with initialization and output methods"""
class System(catacomb.System):
    def __init__(self):
        # Setting system type annotation
        self.format(Types.TEXT, Types.NUMBER)

        # Initializing torchtext vocabulary
        try: torchtext.vocab._default_unk_index
        except AttributeError:
            def _default_unk_index():
                return 0
            torchtext.vocab._default_unk_index = _default_unk_index

        # Loading saved model upon initialization (no training)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        saved_model = torch.load('./saved_model.pt', map_location=self.device)
        self.model, self.stoi = saved_model['model'], saved_model['stoi']

        # Finalizing system initialization
        self.model.eval()
        self.nlp = en_core_web_sm.load()

    # Implementing `output` interface for type `TEXT -> NUMBER`
    def output(self, sentence):
        # Tokenize string and encode values
        tokenized = [tok.text for tok in self.nlp.tokenizer(sentence)]
        indexed = [self.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(self.device).unsqueeze(1)

        # Perform and return prediction
        length_tensor = torch.LongTensor([len(indexed)])
        prediction = torch.sigmoid(self.model(tensor, length_tensor)).item()
        return prediction