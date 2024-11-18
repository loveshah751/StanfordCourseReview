import ast

from transformers import pipeline
import spacy
from inference.Prompt import get_aspect_prompt
from inference.LLMInference import LLMInference


class AspectBasedExtractor:

    def __init__(self, sentiment_analyzer_model="distilbert-base-uncased-finetuned-sst-2-english", spacy_model="en_core_web_sm"):
        # Load Spacy's English NLP model
        self.nlp = spacy.load(spacy_model)
        # Initialize a BERT-based sentiment analysis pipeline
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_analyzer_model)
        self.chat_gpt = LLMInference()


    def extract_aspects_from_paragraph(self,paragraph):
        aspects = []
        for sentence in paragraph:
            aspects.extend(self.extract_aspects(sentence))
        return aspects

    # Step 1: Extract aspects (nouns and noun phrases)
    def extract_aspects(self,text):
        doc = self.nlp(text)
        aspects = [chunk.text for chunk in doc.noun_chunks]
        return aspects

    # Step 2: Analyze sentiment for each aspect using BERT
    def analyze_aspect_sentiments(self, text, aspects):
        results = {}
        for aspect in aspects:
            # Check if the aspect appears in the text
            if aspect in text:
                # Extract the sentence containing the aspect
                sentences = [sent for sent in text.split('.') if aspect in sent]
                aspect_sentiments = []
                for sentence in sentences:
                    # Get the sentiment of the sentence using the BERT-based model
                    sentiment = self.sentiment_analyzer(sentence.strip())[0]
                    aspect_sentiments.append((sentence.strip(), sentiment))
                results[aspect] = aspect_sentiments
        return results

    # Step 3: Generate an LLM prompt using the query text and the extracted aspects
    def get_topics_discussed(self, pdfs, course_summary_context):
        paragraph_aspects = self.extract_aspects_from_paragraph(pdfs)
        aspect_prompt = get_aspect_prompt(paragraph_aspects, course_summary_context) # Generate an LLM prompt using the query text and the extracted aspects
        llm_response = self.chat_gpt.generate_response(aspect_prompt) # Generate a response using the query text and the extracted aspects
        return ast.literal_eval(llm_response)