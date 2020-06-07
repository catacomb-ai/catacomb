import os
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from system import *

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

system = System()

@app.route("/predict", methods=['POST'])
def predict():
    input_object = request.get_json()['input']
    return jsonify(output=system.output(input_object))

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=int(os.getenv('PORT', 5000)), debug=True)
