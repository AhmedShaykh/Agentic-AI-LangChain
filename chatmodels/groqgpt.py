from langchain_groq import ChatGroq;
from dotenv import load_dotenv;
import os;

load_dotenv();

model = ChatGroq( # Model Class
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
);

response = model.invoke("Hi Groq GPT! how are you?");

print(response.content);