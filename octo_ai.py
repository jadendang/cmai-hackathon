from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter
from langchain_community.embeddings import OctoAIEmbeddings
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Milvus
from langchain_community.document_loaders.csv_loader import CSVLoader
import os

load_dotenv()

OCTOAI_API_TOKEN = os.environ["OCTOAI_API_TOKEN"]

loader = CSVLoader(
    file_path="/Users/jaden/Desktop/hackathon may 2024/cmai-hackathon/testhousing.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["bedroom","baths","fullBaths","stories","city","state","zipCode","countryCode","propertyId","listingId","timeZone","hasVirtualTour","has3dTour","description","price","hoaValue","sqFt","pricePerSqFt","location","yearBuilt"],
    },
)

data = loader.load()

print(data[0])


llm = OctoAIEndpoint(
        model_name="llama-2-13b-chat-fp16",
        max_tokens=1024,
        presence_penalty=0,
        temperature=0.1,
        top_p=0.9,
        
    )
embeddings = OctoAIEmbeddings(endpoint_url="https://text.octoai.run/v1/embeddings")

vector_store = Milvus.from_documents(
    data,
    embedding=embeddings,
    connection_args={"host": "localhost", "port": 19530},
    collection_name="house_listings"
)

retriever = vector_store.as_retriever()

print("context retrieved:")
print(retriever)

template="""You are a document-scanning assistant for question-answering tasks. Listings are given as a csv file with headers as the first row and data values in the rest of the rows. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise but descriptive.
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


print(chain.invoke("Can you provide listings that include a hot tub, was built before 1992, 3 beds, 2 baths, single story, roughly 1800 sq ft. I also want the listings to have a recently remodeled kitchen."))