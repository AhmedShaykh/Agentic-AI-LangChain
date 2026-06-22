from langchain_community.vectorstores import Chroma;
from langchain_mistralai import MistralAIEmbeddings;
from langchain_core.documents import Document;
from dotenv import load_dotenv;
import chromadb;
import os;

load_dotenv();

docs = [ # Create Own Documents
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"})
];

embedding_model = MistralAIEmbeddings();

chroma_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE")
);

vectorstore = Chroma.from_documents(
    collection_name="langchain_vectorstore",
    documents=docs,
    embedding=embedding_model,
    client=chroma_client
);

result = vectorstore.similarity_search("what is used for data analysis?", k=2); # K Is The Number Of Results

for r in result:

    print(r.page_content);

    print(r.metadata);

retriver = vectorstore.as_retriever(); # Retriver (Default Similarity Search)

docs = retriver.invoke("Explain deep learning");

for d in docs:

    print(d.page_content);