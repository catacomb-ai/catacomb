import torch
import spacy

from rnn import RNN
from torchtext import vocab

# stupid vocab loading stuff
try:
    vocab._default_unk_index
except AttributeError:
    def _default_unk_index():
        return 0
    vocab._default_unk_index = _default_unk_index



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
senti_rnn = torch.load('./senti_rnn.pt', map_location=device)

stoi = senti_rnn['stoi']
model = senti_rnn['model']

model.eval()
nlp = spacy.load('en')

def predict_sentiment(sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()
