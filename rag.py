import os
from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from services.chroma_service import ChromaService
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

store_messages_history = {}

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama3-8b-8192")

qa_prompt = (
    "Dado o contexto e histórico fornecidos, responda a pergunta do usuário."
    "Se você não souber a resposta, diga que não sabe."
    "Considere que o usuário é um profissional da área de saúde e que está consultando o material."
    "Seja detalhista na resposta: não evite termos técnicos."
    "Você deve deixar explícito os nomes e páginas dos PDFs do contexto que foram usados."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

document_prompt = PromptTemplate(
    input_variables=["page_content", "title"],
    template='[O texto a seguir foi retirado de "{title}" na página {page}] {page_content}',
)

question_answer_chain = create_stuff_documents_chain(
    llm, qa_prompt, document_prompt=document_prompt
)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store_messages_history:
        store_messages_history[session_id] = ChatMessageHistory()
    return store_messages_history[session_id]


def get_rag_chain():
    retriever = ChromaService().load_retriever()
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain
