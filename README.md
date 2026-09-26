# RAG End-to-End: Sports Nutrition Assistant (Advanced & Evaluation)

This repository aims to explore and master the creation of an advanced RAG (Retrieval-Augmented Generation) pipeline, as well as its evaluation using dedicated frameworks, applied to **Sports Nutrition** guidelines.

## Use Case
Building a precise assistant for coaches, dietitians, or athletes. It must answer complex questions about fueling strategies (e.g., "What should I eat during a 3-hour marathon?") based strictly on nutritional science documents, without hallucinating unsafe advice.

## Planned Features

1. **Advanced RAG**:
   - **Semantic Chunking**: Intelligent splitting based on meaning rather than character count (crucial so we don't cut a recipe or supplement protocol in half).
   - **Hybrid Search**: Combination of lexical search (BM25 - great for specific supplements or vitamins) and vector search (dense).
   - **Query Expansion**: Improving the user query via an LLM before searching.
   - **Re-ranking**: Re-ordering results by a dedicated model (e.g., Cohere Rerank) to maximize precision.

2. **Evaluation** (Ragas, TruLens, DeepEval):
   - **Groundedness / Faithfulness**: Ensuring the LLM does not hallucinate dosages or macronutrient ratios and relies purely on the provided context.
   - **Answer Relevance**: Verifying that the answer matches the question.
   - **Context Precision & Recall**: Evaluating the quality of retrieved documents.

## Installation

1. Create a virtual environment: `python -m venv .venv`
2. Activate the environment: `.\.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux)
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your API keys.

## Quickstart (Phase 1)
Run `main.py` to test the baseline RAG system (Naive chunking and simple vector search).
