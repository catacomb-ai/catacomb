from flask import Flask, request
from system import *

app = Flask(__name__)
model = Model()

@app.route("/predict", methods=['POST'])
def predict():
    input_text = request.get_json()['input']
    return {'output': model.output(input_text)}

if __name__ == "__main__":
	app.run(port=5000, debug=True)