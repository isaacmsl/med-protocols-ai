from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

instructions = """Avalia a resposta do aluno com base na prova conceitual por similaridade e classifique como verdadeiro ou falso:
- False: Sem correspondência conceitual ou similiridade
- True: Quase ou todo conceito é igual e similar
- Key criteria: Conceito de ser igual, não o texto.
"""


class Grade(BaseModel):
    score: float = Field(
        description="O quanto a resposta do aluno é precisa em relação a resposta esperada entre 0 e 1.",
        ge=0.00,
        le=1.00,
        multiple_of=0.01
    )


def accuracy(outputs: dict, reference_outputs: dict) -> bool:
    llm = ChatGroq()
    structured_llm = llm.with_structured_output(Grade)

    messages = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": f"""Resposta esperada: {reference_outputs["answer"]}; 
            Resposta do aluno: {outputs["response"]}""",
        },
    ]
    response = structured_llm.invoke(messages)
    return response.score
