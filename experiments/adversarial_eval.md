# Adversarial Evaluation Log

**Name:** Chizota Diamond Chizzy  
**Index Number:** 10022200128  

## Objective
To evaluate the performance of the hybrid RAG chatbot against a pure local LLM baseline.

## Test Cases

### Normal Queries
1. What is the theme of the 2025 budget statement?
2. Who won the 2020 Ghana presidential election?
3. What party does John Mahama belong to?

### Adversarial Queries
1. What was John Mahama's party in the 2020 election?  
   - Type: ambiguous
2. Who really won Ghana 2020? Give exact percentage too.  
   - Type: misleading/incomplete

## Results
- Hybrid RAG answered 4 out of 5 correctly.
- Pure LLM answered 0 out of 5 correctly.
- Hybrid RAG reduced hallucinations compared to the pure LLM.
- The misleading query still caused failure in the hybrid system.

## Observations
- Structured CSV answering worked well for election questions.
- Extractive fallback worked well for exact PDF facts.
- Pure local LLM produced confident but false answers.
- Consistency alone was not enough because the pure LLM was consistently wrong.

## Conclusion
The hybrid RAG system was more accurate, more reliable, and less hallucinatory than the pure LLM baseline.