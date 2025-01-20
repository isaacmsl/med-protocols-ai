from langsmith import Client

from tests.target import target
from tests.evaluator import accuracy


client = Client()

experiment_results = client.evaluate(
    target,
    data="PCDT Perguntas e Respostas",
    evaluators=[
        accuracy,
    ],
    max_concurrency=1,
)
