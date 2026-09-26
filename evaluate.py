import os
from dotenv import load_dotenv
from openai import OpenAI
from main import setup_rag, ask_question 

load_dotenv()
client = OpenAI()

test_questions = [
    "What is the recommended protein intake for post-workout recovery?",
    "Can I eat 100g of fat before a workout?",
    "How much water should I drink before exercise?"
]

def judge_faithfulness(question: str, context: str, answer: str) -> dict:
    prompt = f"""You are an impartial judge evaluating a RAG system.
Given:
- CONTEXT: {context}
- ANSWER: {answer}
Task: Evaluate if the ANSWER is faithful to the CONTEXT (no hallucinations).
Respond with ONLY a JSON object:
{{"score": <float 0.0 to 1.0>, "reason": "<brief explanation>"}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def judge_answer_relevance(question: str, answer: str) -> dict:
    prompt = f"""You are an impartial judge evaluating a RAG system.
Given:
- QUESTION: {question}
- ANSWER: {answer}
Task: Evaluate if the ANSWER directly and concisely addresses the QUESTION.
Respond with ONLY a JSON object:
{{"score": <float 0.0 to 1.0>, "reason": "<brief explanation>"}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def main():
    print("=" * 60)
    print("   EVALUATION DU VRAI RAG EN TEMPS REEL")
    print("=" * 60)

    print("Démarrage du système RAG (Semantic Chunking)...")
    retriever, rag_chain = setup_rag()

    for i, question in enumerate(test_questions):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Question : {question}")
        
        answer, context = ask_question(question, retriever, rag_chain)
        print(f"RAG Answer : {answer}")

        faith_result = judge_faithfulness(question, context, answer)
        print(f"  Faithfulness:      {faith_result}")

        relevance_result = judge_answer_relevance(question, answer)
        print(f"  Answer Relevance:  {relevance_result}")

if __name__ == "__main__":
    main()
