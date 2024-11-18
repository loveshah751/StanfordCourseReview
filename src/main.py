import ast

from PreProcessor import PreProcessor
from inference.LLMInference import LLMInference
from inference.Prompt import get_llm_prompt_for_query, get_aspect_prompt


class MainRunner:

    def __init__(self):
        self.chat_gpt = LLMInference()
        self.preprocessor = PreProcessor()
        self.available_topics = self.get_topics_discussed()

    def generate_system_prompt(self, query: str = "what is RAG, explain with code", topic: str = "Generative AI"):
        code_embeddings, pdf_embeddings = self.preprocessor.pre_process()
        sentence_transformers = self.preprocessor.embeddingTransformer

        pdfs = pdf_embeddings.pdfs
        paragraph_embeddings_indexes = pdf_embeddings.index

        code_snippet = code_embeddings.code_snippets
        code_embeddings_indexes = code_embeddings.index

        code_result = sentence_transformers.search_query_from_python_code_with_faiss(query, code_embeddings_indexes, code_snippet, 5)
        text_results = sentence_transformers.search_with_faiss(query, paragraph_embeddings_indexes, pdfs, 5)
        # if text_results and code_result: are none then return Sorry I am not able to find the answer
        query_prompt = get_llm_prompt_for_query(query, text_results, code_result, topic)
        return query_prompt

    def get_topics_discussed(self):
        system_query = "what is this course about"
        query_prompt = self.generate_system_prompt(system_query)
        course_summary = self.chat_gpt.generate_response(query_prompt)
        pdf_documents = self.preprocessor.pdf_documents
        return self.preprocessor.aspectBasedExtractor.get_topics_discussed([item for sublist in pdf_documents for item in sublist], course_summary)