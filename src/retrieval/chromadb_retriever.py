import chromadb
from typing import List, Dict, Optional

class ChromaDBRetriever:
    def __init__(self, chroma_collection: chromadb.Collection, preprocessor):
        self.collection = chroma_collection
        self.preprocessor = preprocessor

    def query(self, question: str, document_filters: Optional[List[str]] = None, topk: int = 10) -> Dict[str, float]:
        question = question.lower()
        question = self.preprocessor.preprocess(question)

        kwargs = {
            "query_texts": [question],
            "n_results": topk
        }

        if document_filters:
            if isinstance(document_filters, str):
                document_filters = [document_filters]
            filter_dict = {"document": {"$in": document_filters}}
            kwargs["where"] = filter_dict
            
        results = self.collection.query(**kwargs)
        
        doc_ids = results['ids'][0] if results['ids'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        return {doc_id: -distance for doc_id, distance in zip(doc_ids, distances)}
