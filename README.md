# Trained Model Examples
In this repository, we serve several example trained models to make predictions over HTTP.

## Examples
- Sentiment Classifier (PyTorch)
- MNIST Digit Classifier (PyTorch Lightning)
- Fine-Tuned Language Model for Sentiment Classification (HuggingFace)
- Fine-Tuned Language Model for Question Answering (HuggingFace)
- Fine-Tuned Language Model for Summarization (HuggingFace)


## Usage
Install `catacomb` by running:

```
pip install catacomb-ai
```

For each example, run `catacomb` in the example directory to deploy the machine learning instance to Catacomb.

## Architecture
In each example directory, custom machine learning system code is defined in `system.py`, where we only load a pre-trained model (i.e. no training occurs within this application). In particular, we implement the `output()` interface on Catacomb's `System` class, which is called in our generated `server.py` file and `Dockerfile`.

## License
MIT
