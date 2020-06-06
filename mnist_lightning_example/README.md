# Image Recognition Example
> Image Recognition using PyTorch Lightning

In this example, we serve a trained Feed Forward Neural Network to make predictions over HTTP.

## Architecture
Custom machine learning system code is defined in `system.py`, where we only load a pre-trained model (i.e. no training occurs within this application). In particular, we implement the `output()` interface on Catacomb's `Model` class, which is called in our generated `server.py` file.

## Usage
1. Install dependencies with [Pipenv](https://pipenv-fork.readthedocs.io/en/latest/) using `pipenv install`.
2. Run server using `pipenv run python server.py`.

## Data
We have provided a few example images from the MNIST dataset of handwritten digits in the `MNIST/testing/` folder.

### Requests and Responses
You can `curl` the HTTP server to make a prediction:

```bash
curl --header "Content-Type: application/json" \
    --request POST \
    --data '{"input" : "'"$( base64 ./MNIST/testing/4/4.png)"'"}' \
    http://localhost:5000/predict
```

Note that the image is base64 encoded when sending a request.

The server should return a JSON object representing an integer value of the models prediction of the given digit:

```javascript
{
  "output": 4
}
```

## License
MIT
