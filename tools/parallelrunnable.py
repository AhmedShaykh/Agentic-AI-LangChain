from langchain_core.runnables import RunnableParallel, RunnableLambda;
from langchain_core.output_parsers import StrOutputParser;
from langchain_core.prompts import ChatPromptTemplate;
from langchain_mistralai import ChatMistralAI;
from dotenv import load_dotenv;

load_dotenv();

model = ChatMistralAI(model="mistral-small-2506");

parser = StrOutputParser();

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} In 5 6 Lines"
);

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} In Detail"
);

chain = RunnableParallel({
    "short" : RunnableLambda(lambda x: x["short"]) | short_prompt | model | parser, # Pipeline
    "detailed" : RunnableLambda(lambda x: x["detailed"]) | detailed_prompt | model |parser
});

result = chain.invoke({
    "short" : { "topic" : "Generative AI" },
    "detailed" : { "topic" : "Agentic AI" }
});

print(f"\n Result: {result}");

print(f"\n Short: {result["short"]}");

print(f"\n Detail: {result["detailed"]}");