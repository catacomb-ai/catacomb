"""1) Include/define any dependencies for catacomb.System class"""
from transformers import pipeline
import catacomb

MODEL_NAME = "t5-small"

"""2) Implementing catacomb.System class with initialization and output methods"""
class System(catacomb.System):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        self.summarization_pipeline = pipeline(
            "summarization",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME
        )

    def output(self, input):
        return self.summarization_pipeline(sentence)
