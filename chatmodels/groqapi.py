from langchain_groq import ChatGroq;
from dotenv import load_dotenv;
import os;

load_dotenv();

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.6, # Better Result Or Creative Response
    max_tokens=30 # Only 20 Words Generate (Keywords Limit)
);

response = model.invoke("Hi Groq! how are you?");

print(response.content);