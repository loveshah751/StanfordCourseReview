import os
import re
from pdfminer.high_level import extract_text


class DocumentLoader:

    def __init__(self, file_dir,code_dir):
        self.pdf_dir = file_dir
        self.code_dir = code_dir


    def process_pdf_documents(self, window_size, step_size):
        print("Processing PDF documents...")
        paragraphs = []
        for file in os.listdir(self.pdf_dir):
            if file.endswith(".pdf"):
                fname = os.path.join(self.pdf_dir, file)
                paragraphs.append(process_pdf_document(fname, window_size, step_size))
        return paragraphs

    def load_python_files(self):
        print("Processing Python scripts...")
        code_snippets = []
        for file in os.listdir(self.code_dir):
            if file.endswith(".py"):
                file_path = os.path.join(self.code_dir, file)
                code_snippets.append(process_python_script_file(file_path))
        return code_snippets

def process_pdf_document(fname, window_size, step_size):
    text = extract_text(fname)
    text = " ".join(text.split())
    text = clean_text(text)
    text_tokens = text.split()

    sentences = []
    for i in range(0, len(text_tokens), step_size):
        window = text_tokens[i : i + window_size]
        sentences.append(window)
        if len(window) < window_size:
            break

    paragraphs = [" ".join(s) for s in sentences]
    return paragraphs

def process_python_script_file(file_path):
    with open(file_path, "r") as file:
        code_snippet = file.read()
    #cleaned_code_text = clean_text(code_snippet)
    return split_code_into_segments(code_snippet)

def clean_text(text):
    #Remove special characters and punctuation using regex, keep only alphanumeric characters
    cleaned_text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    # Convert to lowercase
    cleaned_text = cleaned_text.lower()
    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def split_code_into_segments(code):
    # Use a simple regex to split code into functions or classes
    segments = re.split(r'\n(?=def |class )', code)
    return segments

def indexing_already_present(index_file):
    return os.path.exists(index_file)
