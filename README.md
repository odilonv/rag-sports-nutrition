# AI Sports Nutritionist (Advanced RAG Pipeline)

**Author:** Odilon VIDAL

## Overview

This project is an advanced Retrieval-Augmented Generation (RAG) system specialized in sports nutrition. It acts as an AI sports nutritionist capable of answering complex queries based on specific medical and nutritional guidelines, while actively preventing AI hallucinations.

## Architecture

The pipeline is built with a highly optimized, enterprise-grade architecture:

```text
User Query
    |
    v
[ Query Rewriter (LLM) ]  <- Transforms vague inputs into precise search queries
    |
    v
[ Hybrid Retriever                              ]
  [ BM25 (50%)  |  ChromaDB + Embeddings (50%) ]  <- Keywords + Semantic search
    |
    v
[ Cohere Reranker (top 2) ]  <- Keeps only the best documents
    |
    v
[ gpt-4o-mini (LLM) ]  <- Generates the final grounded answer
```

## Key Components

| Component           | Technology                   | Purpose                                         |
| ------------------- | ---------------------------- | ----------------------------------------------- |
| **Chunking**        | SemanticChunker              | Splits text by meaning, not by character limits |
| **Vector Store**    | ChromaDB + OpenAI Embeddings | Semantic similarity search                      |
| **Keyword Search**  | BM25                         | Exact-match keyword retrieval                   |
| **Hybrid Fusion**   | EnsembleRetriever (RRF)      | Combines BM25 and vector results                |
| **Reranking**       | Cohere rerank-english-v3.0   | Cross-encoder reranking (top 2)                 |
| **Query Rewriting** | gpt-4o-mini                  | Transforms vague queries into precise ones      |
| **Generation**      | gpt-4o-mini                  | Final answer generation                         |
| **UI**              | Streamlit                    | Chat interface with context transparency        |
| **Evaluation**      | DeepEval + custom evaluator  | LLM-as-a-judge automated testing                |

## Automated Evaluation

The system is rigorously tested using two strategies:

### 1. Custom LLM-as-a-Judge (`evaluate.py`)

A hand-crafted evaluator using `gpt-4o-mini` as judge:

- **Faithfulness** - Does the answer stay within the retrieved context? (no hallucination)
- **Answer Relevancy** - Does the answer directly address the user question?

### 2. DeepEval Framework (`test_deepeval.py`)

A professional testing suite (the PyTest for AI), with calibrated metrics and rich terminal reporting.

> **Current Pass Rate: 100%** (Faithfulness: 1.0 | Answer Relevancy: 1.0)

## Project Structure

```text
rag-sports-nutrition/
|-- data/
|   `-- sports_nutrition_guidelines.txt  # Source knowledge base
|-- app.py              # Streamlit chat interface
|-- main.py             # RAG pipeline (setup_rag + ask_question)
|-- evaluate.py         # Custom LLM-as-a-judge evaluator
|-- test_deepeval.py    # DeepEval automated test suite
|-- requirements.txt
|-- .env                # API keys (not committed to Git)
`-- .gitignore
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file at the root of the project:

```env
OPENAI_API_KEY=your_openai_key_here
COHERE_API_KEY=your_cohere_key_here
```

### 3. Launch the Streamlit chat interface

```bash
streamlit run app.py
```

### 4. Run automated evaluation tests

```bash
# Custom evaluator
python evaluate.py

# DeepEval framework
deepeval test run test_deepeval.py
```

## Tech Stack

- **Python 3.13**
- **LangChain** (LCEL, LangChain Classic, LangChain Community)
- **OpenAI** (gpt-4o-mini, text-embedding-3-small)
- **ChromaDB** (local vector store)
- **Cohere** (rerank-english-v3.0)
- **DeepEval** (LLM evaluation framework)
- **Streamlit** (web UI)