# Trained Model Example
> Sentiment Analysis using PyTorch

In this repository, we serve a trained Recurrent Neural Network (LSTM) to make predictions over HTTP.

## Usage
1. Install dependencies using [Pipenv](): `pipenv install`.
2. Run server: `pipenv run python interface.py`.

### Requests and Responses
You can `curl` the HTTP server to make a prediction:

```bash
curl --header "Content-Type: application/json" \
    --request POST \
    --data '{"input": "Life sucks and I hate it here!"}' \
    http://localhost:5000/predict
```

The server should return a JSON object representing a continuous value between 0 and 1:

```javascript
{
  "output": 0.06209390237927437
}
```