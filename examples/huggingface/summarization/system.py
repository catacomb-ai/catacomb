"""1) Include/define any dependencies for catacomb.System class"""
from transformers import pipeline
import catacomb

MODEL_NAME = "t5-small"

"""2) Implementing catacomb.System class with initialization and output methods"""
class System(catacomb.System):
    def __init__(self):

        self.summarization_pipeline = pipeline(
            "summarization",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME
        )

    def output(self, input):
        text = input['text']
        min_length, max_length = input['min_length'], input['max_length']
        return self.summarization_pipeline(text, min_length=min_length, max_length=max_length)[0]
