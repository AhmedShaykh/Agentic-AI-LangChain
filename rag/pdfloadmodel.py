from langchain_community.document_loaders import PyPDFLoader;
from langchain_core.prompts import ChatPromptTemplate;
from langchain_groq import ChatGroq;
from dotenv import load_dotenv;

load_dotenv();

data = PyPDFLoader("rag/docs/langchain.pdf");

docs = data.load();

template = ChatPromptTemplate.from_messages([
    ("system", "You are a AI that summarizes the text"),
    ("human", "{data}")
]);

model = ChatGroq(model="llama-3.3-70b-versatile");

prompt = template.format_messages(data = docs[0].page_content);

result = model.invoke(prompt);

print(result.content);