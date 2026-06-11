from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

class SbertEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "bkai-foundation-models/vietnamese-bi-encoder"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        # Create embeddings for documents
        embeddings = self.model.encode(input, convert_to_tensor=True)
        return embeddings.cpu().numpy().tolist()  # Return as list of lists for ChromaDB compatibility
