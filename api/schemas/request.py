from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_classifier: bool = True
    use_reranker: bool = True
