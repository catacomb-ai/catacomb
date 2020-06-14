# Sentiment Analysis Example
> Sentiment Analysis using PyTorch

In this example, we serve a trained Recurrent Neural Network (LSTM) to make predictions through the Catacomb application.

## Architecture
Custom machine learning system code is defined in `system.py`, where we only load a pre-trained model (i.e. no training occurs within this application). In particular, we implement the `output()` interface on Catacomb's `System` class. The Catacomb server makes calls to this class to generate predictions.

## Usage
1. Install the Catacomb CLI using `pip install catacomb-ai`
2. Download [Docker Desktop](https://www.docker.com/products/docker-desktop) and sign in.
3. Run `catacomb` in the directory with `system.py`

## License
MIT
