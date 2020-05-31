import torch
import en_core_web_sm
from flask import Flask, request
from torchtext import vocab
from model import RNN

# Model Set-Up and Helper Function Definitions
def predict_sentiment(model, sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()

try:
    vocab._default_unk_index
except AttributeError:
    def _default_unk_index():
        return 0
    vocab._default_unk_index = _default_unk_index

# Model Instantiation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
saved_model = torch.load('./saved_model.pt', map_location=device)
model, stoi = saved_model['model'], saved_model['stoi']

model.eval()
nlp = en_core_web_sm.load()

# Server Instantiation
app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
	return {'output': predict_sentiment(model, request.get_json()['input'])}

if __name__ == "__main__":
	app.run(port=5000, debug=True)