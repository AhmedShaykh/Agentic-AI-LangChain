from langchain_classic.retrievers.multi_query import MultiQueryRetriever;
from langchain_google_genai import GoogleGenerativeAIEmbeddings;
from langchain_mistralai import ChatMistralAI;
from langchain_chroma import Chroma;
from dotenv import load_dotenv;
import chromadb;
import os;

load_dotenv();

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
);

chroma_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE")
);

vectorstore = Chroma(
    collection_name="langchain_retrieverembedding",
    embedding_function=embedding_model,
    client=chroma_client
);

similarity_retriever = vectorstore.as_retriever(
    search_type="similarity", # Default
    search_kwargs={"k": 3} # Top 3 Results
);

print("\nSimilarity Search Results\n");

similarity_docs = similarity_retriever.invoke("What Is Gradient Descent?");

for doc in similarity_docs:

    print(doc.page_content);

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
);

print("\nMaximal Marginal Relevance Results\n");

mmr_docs = mmr_retriever.invoke("What Is Gradient Descent?");

for doc in mmr_docs:

    print(doc.page_content);

llm = ChatMistralAI(model="mistral-small-latest");

base_retriever = vectorstore.as_retriever();

multi_query_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm); # Multi Query

print("\nMulti-Query Retriever Results\n");

mq_docs = multi_query_retriever.invoke("What Is Gradient Descent?");

for doc in mq_docs:

    print(doc.page_content);