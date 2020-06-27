"""1) Include/define any dependencies for catacomb.System class"""
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import catacomb

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

"""2) Implementing catacomb.System class with initialization and output methods"""
class System(catacomb.System):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer
        )

    def output(self, sentence):
        return self.sentiment_pipeline(sentence)[0]
