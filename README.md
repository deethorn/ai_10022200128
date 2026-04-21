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

## test questions
**- Who won the 2020 Ghana presidential election?**
This is a direct factual CSV question and should test whether the system can retrieve and answer a simple structured query correctly.

**- What party did the winner of the 2020 Ghana presidential election belong to?**
This checks whether the system can retrieve related structured evidence and connect the winner to the correct party.

**- What does the 2025 Budget Statement say about the energy sector?**
This is a direct PDF-based question and should test chunk retrieval and grounded summarization from the budget document.

**- Who won?**
This is an ambiguous query to check if the system correctly infers from recent context in a conversation or asks for clarification rather than guessing.

**-Did the 2025 Budget Statement remove all taxes for Ghanaians?**
This is a misleading query to check if it agrees blindly or answers only from the retrieved evidence.

## Author
Chizota Diamond Chizzy  
10022200128
