# Image Recognition Example
> Image Recognition using PyTorch Lightning

In this example, we serve a trained Feed-Forward Neural Network (using
the [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) library) to make through the Catacomb application.

## Architecture
Custom machine learning system code is defined in `system.py`, where we only load a pre-trained model (i.e. no training occurs within this application). In particular, we implement the `output()` interface on Catacomb's `System` class. The Catacomb server then makes calls to this system in order to generate predictions.

## Usage
1. Install the Catacomb CLI with `pip install catacomb-ai`.
2. Download [Docker Desktop](https://www.docker.com/products/docker-desktop) and sign in.
3. Run `catacomb` in the folder containing `system.py`.

## Data
We have provided a few example images from the MNIST dataset of handwritten digits in the `MNIST/testing/` folder.

## License
MIT
