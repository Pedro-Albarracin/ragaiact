import os
from operator import itemgetter
from typing import TypedDict
from dotenv import load_dotenv
import uuid

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import get_buffer_string

load_dotenv()

# Generar un session ID y almacenarlo en una variable
sessionid = str(uuid.uuid4())

# Initialize a LangChain object for retrieving information from Pinecone.
knowledge = PineconeVectorStore.from_existing_index(
    index_name="aiact",
    namespace="aiactrag",
    embedding=OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
)

template = """
Your task is to generate a comprehensive and detailed response by retrieving 
relevant information and augmenting it with generated content. 
The focus is on providing accurate, context-rich answers to the user query, 
with an emphasis on clarity and depth.

Original question:
{context}

Question: {question}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0, model='gpt-4o', streaming=True)


class RagInput(TypedDict):
    question: str
    #sessionId: str  # Añadir sessionId aquí


multiquery = MultiQueryRetriever.from_llm(
    retriever=knowledge.as_retriever(),
    llm=llm,
)


first_chain = (
        RunnableParallel(
            context=(itemgetter("question") | multiquery),
            question=itemgetter("question")
        ) |
        RunnableParallel(
            answer=(ANSWER_PROMPT | llm),
            docs=itemgetter("context")
        )
).with_types(input_type=RagInput)

postgres_memory_url = "postgresql+psycopg://postgres:postgres@rag-aiact-bd.cdqy2mwqmm6i.us-east-1.rds.amazonaws.com:5432/aiact_rag_history?sslmode=require"

get_session_history = lambda: SQLChatMessageHistory(
    connection_string=postgres_memory_url,
    session_id=sessionid,
    async_mode=True
)

template_with_history="""
Given the following conversation and a follow
up question, rephrase the follow up question
to be a standalone question, in its original
language

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

standalone_question_prompt = PromptTemplate.from_template(template_with_history)


standalone_question_mini_chain = RunnableParallel(
    question=RunnableParallel(
        question=RunnablePassthrough(),
        chat_history=lambda x:get_buffer_string(x["chat_history"])
    )
    | standalone_question_prompt
    | llm
    | StrOutputParser()
)


final_chain = RunnableWithMessageHistory(
    runnable=standalone_question_mini_chain | first_chain,
    input_messages_key="question",
    history_messages_key="chat_history",
    output_messages_key="answer",
    get_session_history=get_session_history,
)