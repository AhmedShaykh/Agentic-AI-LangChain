from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint;
from dotenv import load_dotenv;
import os;

load_dotenv();

os.environ["HUGGINGFACEHUB_API_TOKEN"];

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    max_new_tokens=50
);

model = ChatHuggingFace(llm=llm);

response = model.invoke("Hi Deepseek! how are you?");

print(response.content);