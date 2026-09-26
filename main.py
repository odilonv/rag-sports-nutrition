from langchain_community import document_loaders
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.text_splitter import SemanticChunker




# Load environment variables (OpenAI API Key)
load_dotenv()

def setup_rag():
    print("--- Starting Ingestion (Basic RAG) ---")
    
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        print("Error: Please add your OPENAI_API_KEY to the .env file")
        return

    # Load the sports nutrition document
    loader = TextLoader("data/sports_nutrition_guidelines.txt", encoding="utf-8")
    docs = loader.load()
    print(f"Document loaded: {len(docs)} page(s).")

     # Create Chunking, Embeddings and Vector Database
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    text_splitter = SemanticChunker(embeddings)
    chunks = text_splitter.split_documents(docs)
    print(f"Number of chunks generated: {len(chunks)}")

    
    # In-memory ChromaDB for simplicity
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="sports_nutrition_data"
    )

    # Search Engine (Retriever)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Generation
    print("\n--- Generation Phase (LLM) ---")

    llm = ChatOpenAI(model='gpt-4o')
    template = """You are an expert sports nutritionist. Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, rag_chain

def ask_question(query, retriever, rag_chain):
    retrieved_docs = retriever.invoke(query)
    context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)

    answer = rag_chain.invoke(query)

    return answer, context_str


if __name__ == "__main__":
    retriever, rag_chain = setup_rag()

    query= "What is the recommended protein intake for post-workout recovery?"
    answer, context = ask_question(query, retriever, rag_chain)

    print(f"\nRAG Response: \n{answer}\n")
