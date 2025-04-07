     import os
os.environ["RAGAS_DO_NOT_TRACK"] = "true"


import pandas as pd
from ragas import EvaluationDataset, SingleTurnSample
from ragas.validation import validate_required_columns
from ragas.metrics import Faithfulness,  AnswerRelevancy, context_precision, context_recall
from ragas.evaluation import evaluate
from datasets import Dataset
from ragas.evaluation import LangchainLLMWrapper
from datasets import load_dataset
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.schema import HumanMessage


class LoggingCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"\n\n======= LLM PROMPT =======\n{prompts[0]}\n")
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        print(f"\n======= LLM RESPONSE =======\n{response.generations[0][0].text}\n")

def RAGA_Evaluator(test_cases,llm_model,embeddings):
    """
    Evaluates the retrieval chain using LLM-as-a-Judge.

    Parameters:
    - retrieval_chain: The retrieval + document chain.
    - llm: The LLM used for evaluation.
    - statement_file: File containing query statements.
    - feature_files: File containing feature vectors andZ corresponding labels.

    Returns:
    - Average LLM evaluation score.
    """
    # evaluator_llm = LangchainLLMWrapper()
    
    eval_dataset = [
        SingleTurnSample(
            user_input=case["query"],  # Should be a string
            reference=case["ground_truth"],  # Should be a string, not a list
            retrieved_contexts=[case["retrieved_context"]],  # Wrapped in a list as required
            response=case["generated_answer"]  # Should be a string
        )
        for case in test_cases
    ]

    eval_dataset = EvaluationDataset(samples = eval_dataset)


    # print((eval_dataset))
    # # eval_dataset = EvaluationDataset(samples=samples)
    # print(f"\n Total Test Cases: {len(test_cases)}")
    # print("\n Running Evaluation...\n")
    # metrics = [context_precision, context_recall]
    # # metrics = [Faithfulness,context_precision, context_recall]
    # # print("RAGAS_DO_NOT_TRACK =", os.environ.get("RAGAS_DO_NOT_TRACK"))

    # print(validate_required_columns(eval_dataset, metrics))
    dataset = load_dataset("explodinggradients/amnesty_qa", "english_v3", trust_remote_code=True)
    amnesty_subset = dataset["eval"].select(range(2))
    amnesty_subset.to_pandas()
    langchain_llm = ChatOllama(base_url="http://localhost:11434",model=llm_model,callbacks=[LoggingCallbackHandler()],timeout=1000)
    # langchain_embeddings = OllamaEmbeddings(model="llama3")

    # print(f"Using LLM:s {llm}")
    result = evaluate(amnesty_subset,
                  metrics=[
                      Faithfulness(),AnswerRelevancy,context_precision, context_recall], llm=langchain_llm,embeddings=embeddings)

    return result
