import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
from typing import List, Tuple

class DenseRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32):
        """
        Initializes the DenseRetriever by loading the tokenizer and pre-trained model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.batch_size = batch_size
        self.documents: List[str] = []
        self.doc_embeddings: np.ndarray = np.array([])
        self.index = None

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies mean pooling on token embeddings using the attention mask.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Converts a list of texts into dense embeddings using mean pooling.
        """
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.model(**inputs)
                pooled_embeddings = self._mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                all_embeddings.append(pooled_embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

    def add_documents(self, documents: List[str]):
        """
        Adds documents to the retriever, encodes them, and builds a FAISS index.
        """
        self.documents = documents
        self.doc_embeddings = self.encode_texts(documents)

        embedding_dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(self.doc_embeddings)

    def query(self, query_text: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Executes a query and retrieves the top-k most relevant documents.
        """
        query_embedding = self.encode_texts([query_text])
        distances, indices = self.index.search(query_embedding, k)
        return [(self.documents[idx], distances[0][i]) for i, idx in enumerate(indices[0])]


# ===============================
# Example usage
# ===============================
if __name__ == "__main__":
    documents = [
        "The cat is sleeping on the couch.",
        "The dog is playing in the park.",
        "The car is parked on the street.",
        "The sun is shining in the blue sky."
    ]

    retriever = DenseRetriever()
    retriever.add_documents(documents)

    sample_query = "Where is the pet that plays outside?"
    top_results = retriever.query(sample_query, k=2)

    print(f"Query: {sample_query}")
    for rank, (doc, dist) in enumerate(top_results, start=1):
        print(f"Result {rank}: {doc} (Distance: {dist:.4f})")
