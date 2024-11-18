import faiss

class FaissIndexing:
    index: faiss.IndexFlatL2

    def __init__(self, vectors, labels):
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype('float32')
        self.labels = labels

    def create_indexing_for_embeddings(self, fileName:str = "index.index"):
            self.index = faiss.IndexFlatL2(self.dimension)
            self.vectors = self.vectors.astype('float32')
            self.index.add(self.vectors)
            faiss.write_index(self.index, fileName)
