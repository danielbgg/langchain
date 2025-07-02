import bs4
import getpass
import os
import json

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from pymongo.errors import OperationFailure

#https://python.langchain.com/docs/tutorials/rag/?_gl=1*7nv85a*_gcl_au*OTEwMjIwNTg1LjE3NTEzOTIxMDE.*_ga*MzA0MzMzMjc3LjE3NTEzOTIxMDI.*_ga_47WX3HKKY2*czE3NTEzOTIxMDEkbzEkZzEkdDE3NTEzOTIxOTIkajU4JGwwJGgw

with open('credentials.json') as f:
    data = json.load(f)
    mongodb_uri = data['mongodb_uri']
    openai_api_key = data['openai_api_key']
    LANGSMITH_TRACING = data['LANGSMITH_TRACING']
    LANGSMITH_ENDPOINT = data['LANGSMITH_ENDPOINT']
    LANGSMITH_API_KEY = data['LANGSMITH_API_KEY']
    LANGSMITH_PROJECT = data['LANGSMITH_PROJECT']

llm = init_chat_model("gpt-4o-mini", model_provider="openai", openai_api_key=openai_api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

# MongoDB setup
mongo_db_name = 'DEMO-RAG'
mongo_coll_name = 'langchain-rag'
mongo_client = MongoClient(mongodb_uri)
mongo_coll = mongo_client[mongo_db_name][mongo_coll_name]
mongo_db_and_coll_path = f"{mongo_db_name}.{mongo_coll_name}"

vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=mongo_coll,
    index_name="langchain1",
    relevance_score_fn="cosine",
)
# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
document_ids = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])