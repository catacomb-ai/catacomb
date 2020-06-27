"""1) Include/define any dependencies for catacomb.System class"""
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import catacomb

MODEL_NAME = "distilbert-base-uncased-distilled-squad"

"""2) Implementing catacomb.System class with initialization and output methods"""
class System(catacomb.System):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

        self.qa_pipeline = pipeline(
            'question-answering',
            model=model,
            tokenizer=tokenizer
        )

    def output(self, input):
        context, question = input['context'], input['question']
        output = self.qa_pipeline(question=question, context=context)
        return {"answer": output["answer"], "score": output["score"]}
