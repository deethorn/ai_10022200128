# Student: Chizota Diamond Chizzy
# Index Number: 10022200128

from transformers import pipeline


class LLMGenerator:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"):
        self.model_name = model_name
        self.generator = pipeline(
            task="text-generation",
            model=model_name
        )

    def generate_answer(self, prompt: str, max_new_tokens: int = 80) -> str:
        """
        Generate an answer locally using a Hugging Face instruct model.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a careful academic assistant. Use only the provided context and answer briefly."
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