# Sentiment Analysis Example
> Sentiment Analysis using HuggingFace

In this example, we serve a distilled Transformer model to make predictions over HTTP.

## Architecture
Custom machine learning system code is defined in `system.py`, where we only load a pre-trained model (i.e. no training occurs within this application). In particular, we implement the `output()` interface on Catacomb's `System` class, which is called in our generated `server.py` file.

## Usage
1. Install dependencies with [Pipenv](https://pipenv-fork.readthedocs.io/en/latest/) using `pipenv install`.
2. Run server using `pipenv run python server.py`.

### Requests and Responses
You can `curl` the HTTP server to make a prediction:

```bash
curl --header "Content-Type: application/json" \
    --request POST \
    --data '{"input": "I love HuggingFace!"}' \
    http://localhost:5000/predict
```

The server should return a JSON object representing a continuous value between 0 and 1:

```javascript
{
  "output": {
    "label": "POSITIVE",
    "score": 0.9999999
  }
}
```

## License
MIT
