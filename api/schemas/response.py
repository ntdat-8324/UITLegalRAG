from pydantic import BaseModel
from typing import List, Optional

class SourceDocument(BaseModel):
    doc_id: str
    context: str
    article: str
    document: str
    score: float
    rerank_score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    processing_time_ms: float
