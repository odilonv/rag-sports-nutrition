import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# ---- Our test dataset ----
# Each entry simulates a RAG output: 
#   - question: what the user asked
#   - context: what the retriever found (the chunks)
#   - answer: what the LLM generated as a response
test_cases = [
    {
        "question": "What is the recommended protein intake for post-workout recovery?",
        "context": "Consume 20-40g of high-quality protein (rich in leucine, e.g., whey protein) alongside carbohydrates to stimulate muscle repair.",
        "answer": "You should consume 20-40g of high-quality protein post-workout.",
        "ground_truth": "20-40g of high-quality protein."
    },
    {
        "question": "Can I eat 100g of fat before a workout?",
        "context": "30-60 minutes before: A small, easily digestible carbohydrate snack. Avoid high-fiber and high-fat foods.",
        # DELIBERATE HALLUCINATION below!
        "answer": "Yes, eating a lot of fat before a workout gives you maximum energy.",
        "ground_truth": "No, you should avoid high-fat foods before a workout."
    },
    {
        "question": "How much water should I drink before exercise?",
        "context": "Baseline hydration: Drink 5-10 ml/kg of body weight 2-4 hours before exercise.",
        "answer": "Drink 5-10 ml per kg of body weight, 2-4 hours before exercise.",
        "ground_truth": "5-10 ml/kg of body weight, 2-4 hours before exercise."
    }
]


def judge_faithfulness(question: str, context: str, answer: str) -> dict:
    """
    FAITHFULNESS (Groundedness):
    Does the answer contain ONLY information found in the context?
    Score 1.0 = perfectly grounded, 0.0 = pure hallucination.
    """
    prompt = f"""You are an impartial judge evaluating a RAG system.

Given:
- CONTEXT (the retrieved documents): {context}
- ANSWER (the LLM's response): {answer}

Task: Evaluate if the ANSWER is faithful to the CONTEXT. 
The answer must ONLY contain information that can be directly inferred from the context.
If the answer adds information not in the context, or contradicts the context, it is NOT faithful.

Respond with ONLY a JSON object:
{{"score": <float between 0.0 and 1.0>, "reason": "<brief explanation>"}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content


def judge_answer_relevance(question: str, answer: str) -> dict:
    """
    ANSWER RELEVANCE:
    Does the answer actually address the question asked?
    Score 1.0 = perfectly relevant, 0.0 = completely off-topic.
    """
    prompt = f"""You are an impartial judge evaluating a RAG system.

Given:
- QUESTION: {question}
- ANSWER: {answer}

Task: Evaluate if the ANSWER is relevant to the QUESTION.
Does it directly address what was asked? Is it concise and on-topic?

Respond with ONLY a JSON object:
{{"score": <float between 0.0 and 1.0>, "reason": "<brief explanation>"}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content


def main():
    print("=" * 60)
    print("   CUSTOM LLM-AS-A-JUDGE EVALUATION")
    print("=" * 60)

    for i, case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Question:     {case['question']}")
        print(f"RAG Answer:   {case['answer']}")
        print(f"Ground Truth: {case['ground_truth']}")

        # Judge #1: Faithfulness
        faith_result = judge_faithfulness(
            case["question"], case["context"], case["answer"]
        )
        print(f"\n  Faithfulness:      {faith_result}")

        # Judge #2: Answer Relevance
        relevance_result = judge_answer_relevance(
            case["question"], case["answer"]
        )
        print(f"  Answer Relevance:  {relevance_result}")


if __name__ == "__main__":
    main()
