from langchain_google_genai import GoogleGenerativeAIEmbeddings;
from langchain_core.prompts import ChatPromptTemplate;
from langchain_mistralai import ChatMistralAI;
from langchain_chroma import Chroma;
from dotenv import load_dotenv;
import chromadb;
import os;

load_dotenv();

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
);

chroma_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE")
);

vectorstore = Chroma(
    collection_name="langchain_vectorembedding",
    embedding_function=embedding_model,
    client=chroma_client
);

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4, # MMR Picks Top 4 From Those 10
        "fetch_k": 10, # Chroma Fetches 10 Docs
        "lambda_mult": 0.5 # Balanced Relevance + Diversity
    }
);

llm = ChatMistralAI(model="mistral-small-2506");

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
                Use ONLY the provided context to answer the question.
                If the answer is not present in the context,
                say: I could not find the answer in the document."""),
    ("human", """Context:
                {context}
                
                Question:
                {question}""")
]);

print("RAG System Is Ready");

print("Press 0 To Exit\n");

while True:

    query = input("You: ");

    if query == "0":

        break;

    docs = retriever.invoke(query);

    context = "\n\n".join([doc.page_content for doc in docs]); # List Comprehension

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    });

    response = llm.invoke(final_prompt);

    print(f"\nAI: {response.content}\n");