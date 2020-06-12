"""1) Include/define any dependencies for catacomb.System class"""
from transformers import pipeline
import catacomb


"""2) Implementing catacomb.System class with initialization and output methods"""
class System(catacomb.System):
    def __init__(self):

        self.sentiment_pipeline = pipeline(
            "sentiment-analysis"
        )

    def output(self, sentence):
        return self.sentiment_pipeline(sentence)
