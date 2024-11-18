from dotenv import load_dotenv

import openai
import os

class LLMInference:

    def __init__(self):
        load_dotenv()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.api_key = os.getenv("OPENAI_API_KEY")

    def generate_response(self, prompt, streaming=False):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            n=1,
            stop=None,
            temperature=0.2,  # higher temperature means more creative or more hallucination
            messages=messages,
            stream=streaming,
        )
        # Extract the generated response from the API response
        return response if streaming else response.choices[0].message["content"]

    @staticmethod
    def extract_llm_streaming_response(streaming_response:str):
        full_response = []
        for chunk in streaming_response:
            chunk_content = chunk.choices[0].delta.get('content', '')
            full_response.append(chunk_content)
        return ''.join(full_response)
