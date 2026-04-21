from transformers import pipeline


class LLMGenerator:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"):
        self.model_name = model_name
        self.generator = pipeline(
            task="text-generation",
            model=model_name
        )

    def generate_answer(self, prompt: str, max_new_tokens: int = 120) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful academic assistant. "
                    "Use only the provided context. "
                    "Answer briefly and directly when the answer is present in the context. "
                    "Do not answer with document titles or generic headings unless that is exactly what the user asked. "
                    "If the answer is not in the context, say exactly: "
                    "'I could not find the answer in the provided documents.'"
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        output = self.generator(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False
        )

        generated = output[0]["generated_text"]

        if isinstance(generated, list):
            return generated[-1]["content"].strip()

        return str(generated).strip()