from langchain_community.document_loaders import WebBaseLoader;
from rich import print;

url = "https://xyfora-website.vercel.app/";

data = WebBaseLoader(url);

docs = data.load();

print(f"\n{docs}");

print(f"\n{docs[0].page_content}");

print("Web Length: ", len(docs));