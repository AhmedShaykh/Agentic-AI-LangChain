from langchain_core.runnables import RunnableParallel, RunnablePassthrough;
from langchain_core.output_parsers import StrOutputParser;
from langchain_core.prompts import ChatPromptTemplate;
from langchain_mistralai import ChatMistralAI;
from dotenv import load_dotenv;

load_dotenv();

model = ChatMistralAI(model="mistral-small-2506");

parser = StrOutputParser();

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You Are A Code Generator"),
    ("human", "{topic}")
]);

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You Are A Helpful Assistant Who Explains Code In Simple Terms"),
    ("human", "Explain The Following Code In Simple Words:\n{code}")
]);

seq = code_prompt | model | parser;

seq2 = RunnableParallel({
    "code" :  RunnablePassthrough(), # Same Thing Pass As Same Thing Return (Code)
    "explanation" : explain_prompt | model | parser
});

chain = seq | seq2;

result = chain.invoke({
    "topic" : "Please Write A Code Of Lamdba Function Hello Print In Python"
});

print(f"\n Code: {result["code"]}");

print(f"\n Explanation: {result["explanation"]}");