from langchain_google_genai import GoogleGenerativeAIEmbeddings;
from dotenv import load_dotenv;
import os;

load_dotenv();

texts = [
    "I'm Full Stack Developer",
    "I'm Learning Agentic AI"
];

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=40
);

vector = embeddings.embed_documents(texts);

print(f"\n{vector}\n");

print(f"Total Vectors Generated Length: {len(vector)}");

print(f"Dimension Of The First Vector Length: {len(vector[0])}");