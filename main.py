import os
from typing import Annotated

from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse
from langchain_openai import OpenAIEmbeddings

from rag import get_rag_chain

from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_ENDPOINT"] = str(os.getenv("LANGSMITH_ENDPOINT"))
os.environ["LANGSMITH_API_KEY"] = str(os.getenv("LANGSMITH_API_KEY"))
os.environ["LANGSMITH_PROJECT_ID"] = str(os.getenv("LANGSMITH_PROJECT_ID"))

ENDPOINT_QUERY_AI_MAX_SIZE = int(os.getenv("ENDPOINT_QUERY_AI_MAX_SIZE"))
ENDPOINT_EMBEDDING_TEXT_MAX_SIZE = int(os.getenv("ENDPOINT_EMBEDDING_TEXT_MAX_SIZE"))
ENDPOINT_EMBEDDING_MODEL = os.getenv("ENDPOINT_EMBEDDING_MODEL")
ENDPOINT_QUERY_FEEDBACK_CLASSIFY_MAX_SIZE = os.getenv(
    "ENDPOINT_QUERY_FEEDBACK_CLASSIFY_MAX_SIZE"
)
app = FastAPI(title="Bora IA API", version="0.1.0")


@app.get("/ai")
async def ai(
    query: Annotated[str, Query(max_length=ENDPOINT_QUERY_AI_MAX_SIZE)], user_id: str
):
    response = get_rag_chain().invoke(
        {"input": query}, config={"configurable": {"session_id": user_id}}
    )
    return response["answer"]


@app.post("/embedding")
async def embedding(
    text: Annotated[str, Query(max_length=ENDPOINT_EMBEDDING_TEXT_MAX_SIZE)]
):
    embedder = OpenAIEmbeddings(model=ENDPOINT_EMBEDDING_MODEL)
    return embedder.embed_query(text)


@app.get("/ping")
async def root():
    return "Pong! :)"


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")
