#Name: Chizota Diamond Chizzy
#Index Number: 10022200128

import os
from src.data_loader import load_all_documents
from src.cleaner import clean_documents
from src.chunker import chunk_documents
from src.embedder import TextEmbedder
from src.llm_generator import LLMGenerator
from src.evaluator import (
    run_single_rag_test,
    run_single_baseline_test,
    run_consistency_test,
    save_results
)

os.makedirs("logs", exist_ok=True)

docs = load_all_documents()
cleaned_docs = clean_documents(docs)
chunks = chunk_documents(cleaned_docs, strategy="fixed", chunk_size=500, overlap=100)

texts = [chunk["text"] for chunk in chunks]

embedder = TextEmbedder()
chunk_embeddings = embedder.embed_texts(texts)

llm = LLMGenerator(model_name="HuggingFaceTB/SmolLM2-135M-Instruct")

test_cases = [
    {
        "query": "What is the theme of the 2025 budget statement?",
        "expected": "Resetting the Economy for the Ghana We Want",
        "type": "normal"
    },
    {
        "query": "Who won the 2020 Ghana presidential election?",
        "expected": "Nana Akufo Addo of the NPP",
        "type": "normal"
    },
    {
        "query": "What party does John Mahama belong to?",
        "expected": "NDC",
        "type": "normal"
    },
    {
        "query": "What was John Mahama's party in the 2020 election?",
        "expected": "NDC",
        "type": "adversarial_ambiguous"
    },
    {
        "query": "Who really won Ghana 2020? Give exact percentage too.",
        "expected": "Nana Akufo Addo of the NPP",
        "type": "adversarial_misleading"
    }
]

all_results = {
    "rag_results": [],
    "baseline_results": [],
    "consistency_results": []
}

for case in test_cases:
    rag_result = run_single_rag_test(
        query=case["query"],
        expected_answer=case["expected"],
        embedder=embedder,
        chunk_embeddings=chunk_embeddings,
        chunk_docs=chunks,
        llm=llm
    )
    rag_result["case_type"] = case["type"]
    all_results["rag_results"].append(rag_result)

    baseline_result = run_single_baseline_test(
        query=case["query"],
        expected_answer=case["expected"],
        llm=llm
    )
    baseline_result["case_type"] = case["type"]
    all_results["baseline_results"].append(baseline_result)

for case in test_cases[:3]:
    consistency_result = run_consistency_test(
        query=case["query"],
        expected_answer=case["expected"],
        llm=llm,
        embedder=embedder,
        chunk_embeddings=chunk_embeddings,
        chunk_docs=chunks,
        runs=3
    )
    all_results["consistency_results"].append(consistency_result)

save_results(all_results)

print("Evaluation completed.")
print("Results saved to logs/evaluation_results.json")
