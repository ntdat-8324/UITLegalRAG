from rank_bm25 import BM25Plus
import pandas as pd
import pickle

def build_bm25_index(df: pd.DataFrame, k1: float = 1.2, b: float = 0.65) -> BM25Plus:
    """Build BM25Plus index from DataFrame."""
    if 'processed_context_bm25' not in df.columns:
        raise ValueError("DataFrame must contain 'processed_context_bm25'.")
        
    tokenized_corpus = [doc.split() for doc in df['processed_context_bm25'].tolist()]
    bm25 = BM25Plus(tokenized_corpus, k1=k1, b=b)
    return bm25

def save_bm25_index(bm25: BM25Plus, path: str):
    with open(path, 'wb') as f:
        pickle.dump(bm25, f)

def load_bm25_index(path: str) -> BM25Plus:
    with open(path, 'rb') as f:
        return pickle.load(f)
