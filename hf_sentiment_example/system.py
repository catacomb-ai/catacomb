"""1) Include/define any dependencies for catacomb.System class"""
# import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification#, AutoModelForSequenceClassification, AutoTokenizer
import catacomb


"""2) Implementing catacomb.System class with initialization and output methods"""
class System(catacomb.System):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-base-finetuned-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-base-finetuned-sentiment")

        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer
        )

    # Implementing `output` interface for type `TEXT -> LABEL`
    def output(self, sentence):
        return self.sentiment_pipeline(sentence)
