# Academic City RAG Chatbot

**Name:** Chizota Diamond Chizzy  
**Index Number:** 10022200128  
**Repository Name:** ai_10022200128  

## Project Overview
This project is a Retrieval-Augmented Generation (RAG) chatbot built for Academic City using only Python and Streamlit. It answers questions based on two provided datasets:

1. Ghana Election Result CSV
2. 2025 Budget Statement PDF


## Features
- PDF and CSV document loading
- Data cleaning and chunking
- Manual embedding pipeline using sentence-transformers
- Vector retrieval with manual similarity scoring
- FAISS vector storage
- Structured CSV answering for election questions
- Extractive fallback for exact PDF facts
- Local LLM answering with Hugging Face
- Streamlit interface with debug sections
- Evaluation with adversarial queries
- Comparison between RAG and pure LLM baseline

## Project Structure
```text
ai_10022200128/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── src/
├── logs/
├── experiments/
└── docs/
```

## Installation
```bash
pip install -r requirements.txt
```

## Run the App
```bash
streamlit run app.py
```

## Evaluation
Run:
```bash
python run_evaluation.py
```

Evaluation results are saved in:
```text
logs/evaluation_results.json
```

## RAG Pipeline
User Query -> Retrieval -> Context Selection -> Prompt -> LLM -> Response

## Innovation Feature
This project includes a domain-specific hybrid answering system:
- structured CSV answering for election questions,
- extractive fallback for exact PDF facts,
- local LLM fallback for open-ended questions.

This reduced hallucinations and improved factual accuracy.

## Evaluation Summary
- RAG-hybrid accuracy: 80%
- Pure LLM accuracy: 0%
- RAG-hybrid hallucination rate: 20%
- Pure LLM hallucination rate: 100%

## Limitations
The system still struggles with misleading or vague questions if they are not matched by structured rules.

## Author
Chizota Diamond Chizzy  
10022200128