import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict

class CrossEncoderReranker:
    def __init__(self, model_name: str = "itdainb/PhoRanker", preprocessor=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.preprocessor = preprocessor

    def rerank(self, query: str, candidate_docs: List[Dict[str, str]], top_k: int = 5) -> List[Dict[str, str]]:
        if not candidate_docs:
            return []
            
        tokenized_query = query
        if self.preprocessor:
            tokenized_query = self.preprocessor.preprocess(query)
            
        pairs = []
        for doc in candidate_docs:
            # According to user's snippet, we might need to word-segment the context too,
            # but if the context is already preprocessed (from db), we just use it directly.
            # Assuming doc['context'] is the raw context, doc['processed_context'] is segmented.
            context_to_use = doc.get('processed_context', doc.get('context', ''))
            pairs.append([tokenized_query, context_to_use])
            
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=256)
            scores = self.model(**inputs).logits.squeeze(-1).numpy()
            
        # Add scores to candidates
        for i, doc in enumerate(candidate_docs):
            doc['rerank_score'] = float(scores[i])
            
        # Sort and take top_k
        reranked = sorted(candidate_docs, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_k]
