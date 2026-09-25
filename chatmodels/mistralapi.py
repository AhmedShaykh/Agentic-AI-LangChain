from langchain_mistralai import ChatMistralAI;
from dotenv import load_dotenv;
import os;

load_dotenv();

model = ChatMistralAI(
    model="mistral-small-2603",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.6
);

response = model.invoke("Hi Mistral! how are you?");

print(response.content);