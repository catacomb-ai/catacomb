from flask import Flask, request
from system import *

app = Flask(__name__)
model = Model()

@app.route("/predict", methods=['POST'])
def predict():
    input_object = request.get_json()['input']
    return {'output': model.output(input_object)}

if __name__ == "__main__":
	app.run(port=5000, debug=True)