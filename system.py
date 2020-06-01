import torch
import en_core_web_sm
from torchtext import vocab
import torch.nn as nn


# Model definition for later use
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


# Implementing Catacomb's Model class
class Model:
    # Loading saved model upon initialization (no training)
    def __init__(self):
        try:
            vocab._default_unk_index
        except AttributeError:
            def _default_unk_index():
                return 0
            vocab._default_unk_index = _default_unk_index
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        saved_model = torch.load('./saved_model.pt', map_location=self.device)
        self.model, self.stoi = saved_model['model'], saved_model['stoi']

        self.model.eval()
        self.nlp = en_core_web_sm.load()


    # Implementing `output` interface for type `Text -> Number`
    def output(self, sentence):
        tokenized = [tok.text for tok in self.nlp.tokenizer(sentence)]
        indexed = [self.stoi[t] for t in tokenized]
        length = [len(indexed)]
        tensor = torch.LongTensor(indexed).to(self.device)
        tensor = tensor.unsqueeze(1)
        length_tensor = torch.LongTensor(length)
        prediction = torch.sigmoid(self.model(tensor, length_tensor))
        return prediction.item()