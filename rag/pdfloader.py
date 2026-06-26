from langchain_community.document_loaders import PyPDFLoader;
from rich import print;

data = PyPDFLoader("rag/docs/langchain.pdf");

docs = data.load();

print(docs);

print(docs[0].page_content);

print("Docs Length: ", len(docs)); # Per Page Length