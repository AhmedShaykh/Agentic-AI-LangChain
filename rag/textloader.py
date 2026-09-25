from langchain_community.document_loaders import TextLoader;
from rich import print;

data = TextLoader("rag/docs/notes.txt");

docs = data.load();

print(f"\n{docs}");

print(f"\n{docs[0].page_content}");

print("Docs Length: ", len(docs)); # List Length Its Always 1