from langchain_text_splitters import RecursiveCharacterTextSplitter;
from langchain_google_genai import GoogleGenerativeAIEmbeddings;
from langchain_community.document_loaders import PyPDFLoader;
from langchain_chroma import Chroma;
from dotenv import load_dotenv;
import chromadb;
import os;

load_dotenv();

data = PyPDFLoader("rag/docs/langchain.pdf");

docs = data.load();

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 10
);

chunks = splitter.split_documents(docs);

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
    collection_name="langchain_vectorembedding",
    documents=chunks,
    embedding=embedding_model,
    client=chroma_client
);