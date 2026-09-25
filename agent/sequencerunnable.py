from langchain_core.output_parsers import StrOutputParser;
from langchain_core.prompts import ChatPromptTemplate;
from langchain_mistralai import ChatMistralAI;
from dotenv import load_dotenv;

load_dotenv();

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} In Simple Words"
);

model = ChatMistralAI(model="mistral-small-2506");

parser = StrOutputParser(); # Structure Way Represent Output

chain = prompt | model | parser; # Runnable (Old Known As Chain)

result = chain.invoke("Agentic AI");

print(result);