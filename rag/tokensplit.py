from langchain_community.document_loaders import PyPDFLoader;
from langchain_text_splitters import TokenTextSplitter;

data = PyPDFLoader("rag/docs/langchain.pdf");

docs = data.load();

splitter = TokenTextSplitter( # Token Splitter
    chunk_size = 50,
    chunk_overlap=6
);

chunks = splitter.split_documents(docs);

print(f"\nChunks Length: ", len(chunks));

print(f"\n{chunks}\n");

for i, doc in enumerate(chunks):
    
    print(f"{i}: {doc.page_content} \n");