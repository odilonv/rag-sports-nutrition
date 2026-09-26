import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 1. Load environment variables (OpenAI API Key)
load_dotenv()

def main():
    print("--- Starting Ingestion (Basic RAG) ---")
    
    # API Key check
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        print("Error: Please add your OPENAI_API_KEY to the .env file")
        return

    # 2. Load the sports nutrition document
    loader = TextLoader("data/sports_nutrition_guidelines.txt", encoding="utf-8")
    docs = loader.load()
    print(f"Document loaded: {len(docs)} page(s).")

    # 3. Basic chunking (RecursiveCharacterTextSplitter)
    # In v2 we will switch to Semantic Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Number of chunks generated: {len(chunks)}")

    # 4. Create Embeddings and Vector Database
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # In-memory ChromaDB for simplicity
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="sports_nutrition_data"
    )

    # 5. Search Engine (Retriever)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 6. Test a simple query
    query = "What is the recommended protein intake for post-workout recovery?"
    print(f"\nQuery: {query}")
    
    results = retriever.invoke(query)
    print("\nResults retrieved:")
    for i, res in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(res.page_content)
        print("--------------------")

if __name__ == "__main__":
    main()
