from langchain_community.document_loaders import TextLoader;
from langchain_text_splitters import CharacterTextSplitter;

data = TextLoader("rag/docs/notes.txt");

docs = data.load();

splitter = CharacterTextSplitter( # Character Text Splitter
    separator= "", # Ignore New Line ("\n")
    chunk_size = 10,
    chunk_overlap=1
);

chunks = splitter.split_documents(docs);

print(f"\nChunks Length: ", len(chunks));

print(f"\n{chunks}\n");

for i in chunks:

    print(i.page_content);