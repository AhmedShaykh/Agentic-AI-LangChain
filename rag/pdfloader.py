from langchain_community.document_loaders import PyPDFLoader;
from rich import print;

data = PyPDFLoader("rag/docs/langchain.pdf");

docs = data.load();

print(f"\n{docs}");

print(f"\n{docs[0].page_content}");

print("PDF Length: ", len(docs)); # Per Page Length