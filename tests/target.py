import uuid

from rag import get_rag_chain


def target(inputs: str) -> dict:
    user_id = uuid.uuid4().hex

    response = get_rag_chain().invoke(
        {"input": inputs["question"]}, config={"configurable": {"session_id": user_id}}
    )

    return {"response": response["answer"]}
