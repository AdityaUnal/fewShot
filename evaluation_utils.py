import pandas as pd
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness,  AnswerRelevancy, context_precision, context_recall
from ragas.evaluation import evaluate
from langchain.schema import Document
from ragas.llms import LangchainLLMWrapper


def RAGA_Evaluator(test_cases,llm,embeddings):
    """
    Evaluates the retrieval chain using LLM-as-a-Judge.

    Parameters:
    - retrieval_chain: The retrieval + document chain.
    - llm: The LLM used for evaluation.
    - statement_file: File containing query statements.
    - feature_files: File containing feature vectors and corresponding labels.

    Returns:
    - Average LLM evaluation score.
    """
    # evaluator_llm = LangchainLLMWrapper()
    eval_dataset = [
        SingleTurnSample(  # ✅ Using SingleTurnSample (since Instance is missing)
            question=case["query"],
            ground_truths=[case["ground_truth"]],
            contexts=[case["retrieved_context"]],
            answer=case["generated_answer"]
        )
        for case in test_cases
    ]

    print(f"\n Total Test Cases: {len(test_cases)}")
    print("\n Running Evaluation...\n")

    print(f"Using LLM:s {llm}")

    results = evaluate(
            eval_dataset,

            metrics=[
                
                Faithfulness(llm=llm),  # ✅ Use Hugging Face LLM
                AnswerRelevancy(llm=llm,embeddings=embeddings),  # ✅ Use Hugging Face LLM
                context_precision,  # No LLM needed
                context_recall  # No LLM needed
            ]
        )

    print("\n🔹 Evaluation Results:")
    for metric, score in results.items():
        print(f"   {metric}: {score:.4f}")

    return results