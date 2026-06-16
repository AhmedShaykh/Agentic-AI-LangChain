from langchain_google_genai import GoogleGenerativeAIEmbeddings;
from dotenv import load_dotenv;
import os;

load_dotenv();

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=40
);

texts = [
    "Hello! I'm Ahmed Shaykh",
    "I'm Learning LangChain With Embeddings"
];

vector = embeddings.embed_documents(texts);

print(vector);

print(f"Total vectors generated: {len(vector)}");

print(f"Dimension of the first vector: {len(vector[0])}");