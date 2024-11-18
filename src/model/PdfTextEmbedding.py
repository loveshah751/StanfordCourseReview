import faiss


class PdfTextEmbedding:
    index: faiss.IndexFlatL2
    pdfs = []
    def __init__(self, pdfs, text_indexes: faiss.IndexFlatL2):
        self.pdfs = pdfs
        self.index = text_indexes