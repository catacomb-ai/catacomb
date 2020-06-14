# HuggingFace DistilBERT Sentiment Analysis Example
> Sentiment Analysis using a pre-trained HuggingFace DistilBERT Model

In this example, we show how to serve a trained DistilBERT model to make predictions through the Catacomb application.

## Architecture
Custom machine learning system code is defined in `system.py`, where we only load a pre-trained model (i.e. no training occurs within this application). In particular, we implement the `output()` interface on Catacomb's `System` class. The Catacomb server then makes calls to this system in order to generate predictions.

## Usage
1. Install the Catacomb CLI with `pip install catacomb-ai`.
2. Download [Docker Desktop](https://www.docker.com/products/docker-desktop) and sign in.
3. Run `catacomb` in the folder containing `system.py`.

## License
MIT