import faiss
import numpy as np

from loader.DocumentLoader import DocumentLoader, indexing_already_present
from model.CodeSnippetEmbedding import CodeSnippetEmbedding
from embeddings.EmbeddingTransformer import EmbeddingTransformer
from embeddings.FaissIndexing import FaissIndexing
from model.PdfTextEmbedding import PdfTextEmbedding
from AspectBasedExtractor import AspectBasedExtractor

text_indexes_location = "../embeddingIndexes/text.index"
code_indexes_location = "../embeddingIndexes/code.index"

class PreProcessor:
    embeddingTransformer: EmbeddingTransformer
    aspectBasedExtractor: AspectBasedExtractor
    pdf_documents: list
    code_snippets: list
    topics_discussed: list

    def __init__(self):
        print("creating pre-processing instance")
        self.embeddingTransformer = EmbeddingTransformer()
        self.aspectBasedExtractor = AspectBasedExtractor()
        document_loader = DocumentLoader("../documents/pdf/", "../documents/googleCollabNotebook/")
        self.pdf_documents = document_loader.process_pdf_documents( 120, 100)
        self.code_snippets = document_loader.load_python_files()


    def pre_process(self):
        pdf_documents = self.pdf_documents
        code_snippets = self.code_snippets
        #3 create embeddings for the text, code with faiss indexing
        code_indexes, text_indexes  = self.__create_embeddings(pdf_documents, code_snippets)
        return code_indexes, text_indexes

    def __create_embeddings(self, pdf_documents, code_snippets):
        text_indexes = None
        code_indexes = None
        #5 extract the embeddings for the text
        if not indexing_already_present(text_indexes_location):
            text_embeddings = []
            for pdf in pdf_documents:
                text_embeddings.extend(self.embeddingTransformer.get_sentence_embedding(pdf))
            vectors = np.array(text_embeddings)
            faiss_text_indexing = FaissIndexing(vectors, pdf_documents)
            faiss_text_indexing.create_indexing_for_embeddings(text_indexes_location)
            text_indexes = faiss_text_indexing.index
        else:
            text_indexes = faiss.read_index(text_indexes_location)

        #6 extract the embeddings for the code
        if not indexing_already_present(code_indexes_location):
            code_embeddings = []
            for code in code_snippets:
                code_embeddings.extend(self.embeddingTransformer.python_code_embeddings(code))

            code_embeddings = np.array(code_embeddings)
            faiss_code_indexing = FaissIndexing(code_embeddings, code_snippets)
            faiss_code_indexing.create_indexing_for_embeddings(code_indexes_location)
            code_indexes = faiss_code_indexing.index
        else:
            code_indexes = faiss.read_index(text_indexes_location)
        return CodeSnippetEmbedding(code_snippets, code_indexes), PdfTextEmbedding(pdf_documents, text_indexes)

