from langchain_text_splitters import RecursiveCharacterTextSplitter;
from langchain_google_genai import GoogleGenerativeAIEmbeddings;
from langchain_core.documents import Document;
from langchain_chroma import Chroma;
from dotenv import load_dotenv;
import chromadb;
import os;

load_dotenv();

data = [
    Document(page_content="Embeddings convert text into numerical vectors that capture semantic meaning."),
    Document(page_content="Vector databases store embeddings and allow similarity search over high-dimensional vectors."),
    Document(page_content="Retrieval Augmented Generation combines a language model with external knowledge retrieved from a vector database."),
    Document(page_content="RAG systems retrieve relevant documents before sending the context to a large language model."),
    Document(page_content="LangChain provides tools and abstractions for building applications powered by large language models."),
    Document(page_content="LangGraph is a framework for building stateful and multi-step AI agent workflows."),
    Document(page_content="AI agents can use tools such as search, calculators, databases, and APIs to perform tasks."),
    Document(page_content="Tool calling allows a language model to decide when an external function or API should be executed."),
    Document(page_content="Transformers use attention mechanisms to understand relationships between tokens in a sequence."),
    Document(page_content="Large language models generate text by predicting the next token based on the provided context."),
    Document(page_content="Agentic AI systems can plan tasks, execute actions, observe results, and modify their strategy.")
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
    collection_name="langchain_textembedding",
    documents=chunks,
    embedding=embedding_model,
    client=chroma_client
);