from langchain_core.output_parsers import StrOutputParser;
from langchain_core.prompts import ChatPromptTemplate;
from langchain_mistralai import ChatMistralAI;
from langchain_tavily import TavilySearch;
from dotenv import load_dotenv;

load_dotenv();

search_tool = TavilySearch(max_result = 5); # Pre Built In Tool But Require Tavily API

llm = ChatMistralAI(model = "mistral-small-2506");

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant
summarize the following news into clear bullet points
{news}
""");

chain = prompt | llm | StrOutputParser();

news_result = search_tool.run("Latest AI News Of 2026");

result = chain.invoke({"news" : news_result});

print(result);