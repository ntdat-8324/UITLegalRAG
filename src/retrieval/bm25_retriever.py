import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional
import pandas as pd

class BM25Retriever:
    def __init__(self, bm25_model, documents: List[str], ids: List[str], stopword_remover, preprocessor, df: pd.DataFrame, limit: int = 10):
        self.bm25 = bm25_model
        self.documents = documents
        self.ids = ids
        self.stopword_remover = stopword_remover
        self.preprocessor = preprocessor
        self.df = df
        self.limit = limit
        
        self.id_to_pos = {doc_id: idx for idx, doc_id in enumerate(self.ids)}
        
    def query(self, question: str, document_filters: Optional[List[str]] = None, limit: Optional[int] = None) -> Dict[str, float]:
        if limit is None:
            limit = self.limit
            
        question = question.lower()
        question = self.preprocessor.preprocess(question)
        processed_query = self.stopword_remover.remove_stopwords(question)
        tokenized_query = processed_query.split()
        
        scores = self.bm25.get_scores(tokenized_query)
        
        if document_filters:
            if isinstance(document_filters, str):
                document_filters = [document_filters]
                
            filtered_ids = self.df[self.df['document'].isin(document_filters)].index.astype(str).tolist()
            filtered_docs = [self.documents[self.id_to_pos[doc_id]] for doc_id in filtered_ids if doc_id in self.id_to_pos]
            
            if filtered_docs:
                bm25_filtered = BM25Okapi([doc.split() for doc in filtered_docs])
                scores = bm25_filtered.get_scores(tokenized_query)
                top_indices = np.argsort(scores)[::-1][:limit]
                top_ids = [filtered_ids[i] for i in top_indices]
            else:
                top_indices = []
                top_ids = []
        else:
            top_indices = np.argsort(scores)[::-1][:limit]
            top_ids = [self.ids[i] for i in top_indices]
            
        return {doc_id: scores[idx] for idx, doc_id in zip(top_indices, top_ids) if len(scores) > idx}
