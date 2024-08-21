import os
from dotenv import load_dotenv
# from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores.pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "aiact"


loader = PyPDFDirectoryLoader("../pdf/")

docs = loader.load()

print("Cargado el documento en el loader Generando embeddings")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

print("Realizando Split del documento")
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=2048,chunk_overlap=100)

chunks = text_splitter.split_documents(docs)

# Imprime los chunks generados
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")

print("Creando el índice en Pinecone")
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536, # Replace with your model dimensions
        metric="cosine", # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )

print("Guardando los documentos en Pinecone")
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents=chunks,
    index_name=index_name,
    embedding=embeddings, 
    namespace="aiactrag" 
)
