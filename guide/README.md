# Quickstart Guide
> Building machine learning is hard. Using it shouldn't be.

This guide will get you up and running with Catacomb in under 10 minutes. 

We recommend using a *cloud training platform* like [Google Colab](https://colab.research.google.com) to run this demonstration. However, Catacomb fully supports any operating system and runtime, so feel free to use your own computer as well!

This guide doesn't make any assumptions about any underlying model architecture. However,
we will be using a pre-trained model from [Hugging Face](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) as reference for our examples.

## Launching a User Interface

Let's start with a sentiment analysis model - a powerful and fun application of natural language processing. The easiest way to get our model working with Catacomb is to launch a user interface for the existing model locally.

### Installing the Catacomb Library

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

### Implementing the Catacomb Interface

The Catacomb interface requires the implementation of a Python class with two methods.

**`__init__(self)`**

The initialization method is called upon instantiation of the class. This should include neccessary model setup **that only occurs once** such as loading checkpoints, and configuring hyperparameters.

**`output(self, input)`**

Catacomb calls the `output()` method on a given class to perform inferences, given an input argument.

The type of the `input` variable depends on the configuration and implementation of the model. We will cover Catacomb's typing system later, but for now we can implement the interface assuming that each input is well-formatted and in the correct data type.

#### Example

Here's an implemented Catacomb interface for sentiment analysis, where we assume that `input` is a string:

```python
class SentimentClassifier:
    def __init__(self):
        MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        self.sentiment_pipeline = pipeline(
            'sentiment-analysis',
            model=model,
            tokenizer=tokenizer
        )

    def output(self, input):
        return self.sentiment_pipeline(input)[0]
```