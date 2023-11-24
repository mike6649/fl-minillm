import requests
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

prompt = """
We would like to request your feedback on the performance of an AI assistant in response to the user instruction and input displayed above.
Please rate the helpfulness, relevance, accuracy, and level of detail of their response. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please output a single line containing only a value indicating the score for the assistant.
"""

def _metric(pred, model):
    ...


def get_metrics_gpt(predictions, prompts, model="gpt-3.5-turbo"):
    if OPENAI_API_KEY is None:
        return {"gpt-score": None}
    
    gpt_score = 0
    for pred in predictions:
        ...