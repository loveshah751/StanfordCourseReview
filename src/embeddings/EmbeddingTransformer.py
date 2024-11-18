import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer, RobertaModel

class EmbeddingTransformer:
    embeddingModel: SentenceTransformer
    encoder: CrossEncoder

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 crossEncoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 code_bert:str ="microsoft/codebert-base"):
        torch.set_num_threads(1)
        model= SentenceTransformer(model_name)
        self.encoder = CrossEncoder(crossEncoder)
        self.code_bert_model = RobertaModel.from_pretrained(code_bert)
        self.tokenizer = RobertaTokenizer.from_pretrained(code_bert)
        # Use the GPU if available
        if not torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Warning: No GPU found. Please use a GPU for faster inference.")
        else:
            print("GPU Found!")
            model =  model.to('cuda')
        self.embeddingModel = model

    def get_sentence_embedding(self, sentence) -> torch.Tensor:
        return self.embeddingModel.encode(sentence)

    def get_sentence_similarity(self, sentence: str) -> float:
        embeddings = self.get_sentence_embedding(sentence)
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            distance = 1 - similarity
            distances.append(distance)
        return distances

    def encoding(self, sentences, convert_to_tensor=True):
        self.embeddingModel.max_seq_length = 512
        return self.embeddingModel.encode(sentences)

    def python_code_embeddings(self, code_snippets):
        embeddings = []
        for code in code_snippets:
            # Tokenize and convert to input IDs
            inputs = self.tokenizer(code, return_tensors="pt", truncation=True, padding=True)

            # Generate embeddings with CodeBERT
            with torch.no_grad():
                outputs = self.code_bert_model(**inputs)

            # Use the [CLS] token's embedding as the representation
            #cls_embedding = outputs.last_hidden_state.mean(dim=1)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            embeddings.extend(cls_embedding)

        # Stack embeddings into a single tensor
        embeddings = torch.stack(embeddings)
        return embeddings

    def search_query_from_python_code_embeddings(self, query, code_embeddings, code_snippets):
        query_embedding = self.python_code_embeddings([query])
        # Calculate cosine similarity
        similarity_scores = cosine_similarity(query_embedding.mean(dim=1), code_embeddings.mean(dim=1))
        # Sort the results by similarity scores
        top_indices = similarity_scores.argsort()[-3:][::-1][0] # Get top 3 results
        return [code_snippets[idx] for idx in top_indices]

    def search(self, query, embeddings, paragraphs, top_k):
        query_embeddings = self.embeddingModel.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(
            query_embeddings,
            embeddings,
            top_k=top_k)[0]
        cross_input = [[query, paragraphs[hit["corpus_id"]]] for hit in hits]
        cross_scores = self.encoder.predict(cross_input)

        for idx in range(len(cross_scores)):
            hits[idx]["cross_score"] = cross_scores[idx]

        hits = sorted(hits, key=lambda x: x["cross_score"], reverse=True)
        results = []
        return [results.append(paragraphs[hit["corpus_id"]].replace("\n", " ")) for hit in hits[:5]]

    def search_with_faiss(self, query, index, paragraphs, top_k):
        # Encode the query
        query_embeddings = self.encoding(query, convert_to_tensor=False)
        # Ensure the query is in the correct format
        query_embeddings = np.array(query_embeddings).astype('float32').reshape(1, -1)
        # Perform ANN search using Faiss
        distances, indices = index.search(query_embeddings, top_k)
        # Prepare the results based on the indices
        results = []
        for paragraph in paragraphs:
            results.extend([paragraph[idx] for idx in indices[0] if 0 <= idx < len(paragraph)])
        return results

    def search_query_from_python_code_with_faiss(self, query, index, code_snippets, top_k):
        top_k = len(code_snippets) if len(code_snippets) - 1  < top_k else top_k

        query_embedding = self.python_code_embeddings([query]).numpy()
        query_embedding = np.expand_dims(query_embedding[0], axis=0)
        distances, indices =  index.search(query_embedding, top_k)
        # Collect the most similar code snippets based on the search results
        result = []
        for code in code_snippets:
            result.extend([code[idx] for idx in indices[0] if 0 <= idx < len(code)])
        return result

