from langchain_community.document_loaders import WebBaseLoader;
from rich import print;

url = "https://xyfora-website.vercel.app/";

data = WebBaseLoader(url);

docs = data.load();

print(docs);

print(docs[0].page_content);

print("Docs Length: ", len(docs));