
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore

import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


print(PINECONE_API_KEY)

# Change to the specified directory  
os.chdir("D:\GKV chatbot\GKV-Chatbot-using-Llama2-main")
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone

# Initialize Pinecone with your API key
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "gkvchatbot"
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,)

# Load Existing index 

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings)


