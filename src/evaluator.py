#Name: Chizota Diamond Chizzy
#Index Number: 10022200128

import json
from src.prompt_builder import build_baseline_prompt
from src.pipeline import run_rag_pipeline


def normalize_text(text: str) -> str:
    return str(text).strip().lower()


def is_correct(predicted: str, expected: str) -> bool:
    return normalize_text(expected) in normalize_text(predicted) or normalize_text(predicted) in normalize_text(expected)


def detect_hallucination(predicted: str, expected: str) -> int:
    if is_correct(predicted, expected):
        return 0
    return 1


def run_single_rag_test(query, expected_answer, embedder, chunk_embeddings, chunk_docs, llm):
    result = run_rag_pipeline(
        query=query,
        embedder=embedder,
        chunk_embeddings=chunk_embeddings,
        chunk_docs=chunk_docs,
        llm=llm,
        top_k=5
    )

    predicted = result["final_answer"]

    return {
        "query": query,
        "expected_answer": expected_answer,
        "predicted_answer": predicted,
        "method": "rag",
        "answer_source": result["answer_source"],
        "accuracy": 1 if is_correct(predicted, expected_answer) else 0,
        "hallucination": detect_hallucination(predicted, expected_answer)
    }


def run_single_baseline_test(query, expected_answer, llm):
    prompt = build_baseline_prompt(query)
    predicted = llm.generate_answer(prompt)

    return {
        "query": query,
        "expected_answer": expected_answer,
        "predicted_answer": predicted,
        "method": "pure_llm",
        "answer_source": "local_llm_only",
        "accuracy": 1 if is_correct(predicted, expected_answer) else 0,
        "hallucination": detect_hallucination(predicted, expected_answer)
    }


def run_consistency_test(query, expected_answer, llm, embedder, chunk_embeddings, chunk_docs, runs=3):
    rag_answers = []
    llm_answers = []

    for _ in range(runs):
        rag_result = run_rag_pipeline(
            query=query,
            embedder=embedder,
            chunk_embeddings=chunk_embeddings,
            chunk_docs=chunk_docs,
            llm=llm,
            top_k=5
        )
        rag_answers.append(rag_result["final_answer"])

        baseline_prompt = build_baseline_prompt(query)
        llm_answers.append(llm.generate_answer(baseline_prompt))

    rag_unique = len(set([normalize_text(x) for x in rag_answers]))
    llm_unique = len(set([normalize_text(x) for x in llm_answers]))

    return {
        "query": query,
        "expected_answer": expected_answer,
        "rag_answers": rag_answers,
        "pure_llm_answers": llm_answers,
        "rag_consistency_score": 1 / rag_unique,
        "pure_llm_consistency_score": 1 / llm_unique
    }


def save_results(results, filename="logs/evaluation_results.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
