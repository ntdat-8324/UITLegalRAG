from typing import List, Dict, Optional

class HybridRetriever:
    def __init__(self, biencoder, bm25, topk: int = 10, alpha: float = 0.35):
        self.biencoder = biencoder
        self.bm25 = bm25
        self.raw_topk = 50
        self.topk = topk
        self.alpha = alpha

    def scale_scores(self, scores_dict: Dict[str, float]) -> Dict[str, float]:
        if not scores_dict:
            return {}
        values = list(scores_dict.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return {k: 1.0 for k in scores_dict}
        else:
            return {k: (v - min_val) / (max_val - min_val) for k, v in scores_dict.items()}

    def query(self, question: str, document_filters: Optional[List[str]] = None, limit: Optional[int] = None) -> List[str]:
        if limit is None:
            limit = self.topk

        bm25_results = self.bm25.query(question, document_filters=document_filters, limit=self.raw_topk)
        biencoder_results = self.biencoder.query(question, document_filters=document_filters, topk=self.raw_topk)

        bm25_scaled = self.scale_scores(bm25_results)
        biencoder_scaled = self.scale_scores(biencoder_results)

        fused_scores = {}
        all_ids = set(bm25_scaled.keys()).union(set(biencoder_scaled.keys()))

        for doc_id in all_ids:
            bm25_score = bm25_scaled.get(doc_id, 0.0)
            biencoder_score = biencoder_scaled.get(doc_id, 0.0)
            fused_score = ((1 - self.alpha) * bm25_score) + (self.alpha * biencoder_score)
            fused_scores[doc_id] = fused_score

        fused_sorted = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        top_docs = [doc_id for doc_id, score in fused_sorted[:limit]]

        return top_docs
