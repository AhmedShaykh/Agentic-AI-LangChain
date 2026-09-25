from langchain_text_splitters import RecursiveCharacterTextSplitter;
from langchain_community.document_loaders import PyPDFLoader;

data = PyPDFLoader("rag/docs/langchain.pdf");

docs = data.load();

splitter = RecursiveCharacterTextSplitter( # Recursively Text Splitter Based On Meaning
    chunk_size = 100,
    chunk_overlap=10
);

chunks = splitter.split_documents(docs);

print(f"\nChunks Length: ", len(chunks));

print(f"\n{chunks}\n");

for i, doc in enumerate(chunks):
    
    print(f"{i}: {doc.page_content} \n");