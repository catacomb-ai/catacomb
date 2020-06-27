"""1) Include/define any dependencies for catacomb.System class"""
import catacomb
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline


MODEL_NAME = "sshleifer/distilbart-xsum-12-6"

"""2) Implementing catacomb.System class with initialization and output methods"""
class System(catacomb.System):
    def __init__(self):

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelWithLMHead.from_pretrained(MODEL_NAME)

        self.summarization_pipeline = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer
        )

    def output(self, input):
        text = input['text']
        min_length, max_length = input['min_length'], input['max_length']
        return self.summarization_pipeline(text, min_length=min_length, max_length=max_length)[0]
