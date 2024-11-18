import faiss


class CodeSnippetEmbedding:
    index: faiss.IndexFlatL2
    code_snippets = []

    def __init__(self, code_snippets, code_indexes: faiss.IndexFlatL2):
        self.code_snippets = code_snippets
        self.index = code_indexes