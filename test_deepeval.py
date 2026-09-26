from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from main import setup_rag, ask_question

load_dotenv()

retriever, rag_chain = setup_rag()

def test_rag_post_workout():
    
    question = "What is the recommended protein intake for post-workout recovery?"
    
    answer, context = ask_question(question, retriever, rag_chain)

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=[context]
    )

    faithfulness = FaithfulnessMetric(threshold=0.7)
    relevance = AnswerRelevancyMetric(threshold=0.7)

    assert_test(test_case, [faithfulness, relevance])
