from langchain.chat_models import init_chat_model;
from dotenv import load_dotenv;
import os;

load_dotenv();

os.environ["OPENAI_API_KEY"];

model = init_chat_model(
    "gpt-3.5-turbo",
    model_provider="openai"
);

response = model.invoke("Hi OpenAI! how are you?");

print(response.content);