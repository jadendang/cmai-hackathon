from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter
from langchain_community.embeddings import OctoAIEmbeddings
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Milvus
import os

load_dotenv()

OCTOAI_API_TOKEN = os.environ["OCTOAI_API_TOKEN"]

url = "https://en.wikipedia.org/wiki/Star_Wars"

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
    ("div", "Divider")
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

html_header_splits = html_splitter.split_text_from_url(url)

chunk_size = 1024
chunk_overlap = 128
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)

# Split
splits = text_splitter.split_documents(html_header_splits)

llm = OctoAIEndpoint(
        model_name="llama-2-13b-chat-fp16",
        max_tokens=1024,
        presence_penalty=0,
        temperature=0.1,
        top_p=0.9,
        
    )
embeddings = OctoAIEmbeddings(endpoint_url="https://text.octoai.run/v1/embeddings")

vector_store = Milvus.from_documents(
    splits,
    embedding=embeddings,
    connection_args={"host": "localhost", "port": 19530},
    collection_name="starwars"
)

retriever = vector_store.as_retriever()

template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke("Who is Luke's Father?"))