from langchain_community.document_loaders import TextLoader;
from langchain_text_splitters import CharacterTextSplitter;

splitter = CharacterTextSplitter( # Text Splitter
    separator= "", # Ignore New Line ("\n")
    chunk_size = 10,
    chunk_overlap=1
);

data = TextLoader("rag/docs/message.txt");

docs = data.load();

chunks = splitter.split_documents(docs);

print("Chunks Length: ", len(chunks));

print(chunks);

for i in chunks:

    print(i.page_content);