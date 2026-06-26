from langchain_text_splitters import RecursiveCharacterTextSplitter;
from langchain_google_genai import GoogleGenerativeAIEmbeddings;
from langchain_core.documents import Document;
from langchain_chroma import Chroma;
from dotenv import load_dotenv;
import chromadb;
import os;

load_dotenv();

data = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
];

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 10
);

chunks = splitter.split_documents(data);

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
);

chroma_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE")
);

vectorstore = Chroma.from_documents(
    collection_name="langchain_retrieverembedding",
    documents=chunks,
    embedding=embedding_model,
    client=chroma_client
);