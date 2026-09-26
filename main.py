from langchain_community import retrievers
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
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

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
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    base_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    compressor = CohereRerank(top_n=2, model="rerank-english-v3.0")


    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # Generation
    print("\n--- Generation Phase (LLM) ---")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


    rewrite_prompt = ChatPromptTemplate.from_template(
        "Rewrite the following user question into a precise, standalone search query for a sports nutrition database.\n"
        "Just output the rewritten query, nothing else.\n"
        "Question: {question}"
    )
    query_rewriter = rewrite_prompt | llm | StrOutputParser()

    template = """You are an expert sports nutritionist. Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": query_rewriter | retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, rag_chain, query_rewriter

def ask_question(query: str, retriever, rag_chain, query_rewriter):
    better_query = query_rewriter.invoke({"question": query})
    print(f"\nBase : '{query}'")
    print(f"Rewritten  : '{better_query}'")

    context_docs = retriever.invoke(better_query)
    context = "\n".join([doc.page_content for doc in context_docs])
    
    answer = rag_chain.invoke(query)
    
    return answer, context



if __name__ == "__main__":
    retriever, rag_chain, query_rewriter = setup_rag()

    query= "What is the recommended protein intake for post-workout recovery?"
    bad_query = "And what about water?"

    answer, context = ask_question(bad_query, retriever, rag_chain, query_rewriter)

    print(f"\nRAG Response: \n{answer}\n")
