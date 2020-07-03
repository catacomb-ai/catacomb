# Catacomb Guide
> Building machine learning is hard. Using it shouldn't be.

## Quickstart

This guide will get you up and running with Catacomb in under 10 minutes.

We recommend using a *cloud training platform* like [Google Colab](https://colab.research.google.com) to run this demonstration. However, Catacomb fully supports any operating system and runtime, so feel free to use your own computer as well!

### Launching a User Interface

The easiest way to get started with Catacomb is to launch a user interface for an existing local system.

Let's start with a sentiment analysis model - a powerful and fun application of natural language processing. In this guide, we will be relying on a pre-trained model from [Hugging Face](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).

#### Installing the Catacomb Library

The recommended way to install Catacomb is through [Pipenv](https://pipenv-fork.readthedocs.io/en/latest/) (we also support `pip install`), by running:

```
pipenv install catacomb-ai
```

You can confirm your installation was successful by running `catacomb` in your terminal, and seeing a list of commands:

```
Usage: catacomb [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  build
  push
  run
  ```