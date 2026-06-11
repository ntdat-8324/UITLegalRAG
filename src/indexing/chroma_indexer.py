import pandas as pd
import chromadb
from typing import Callable, Any

def build_chroma_index(
    df: pd.DataFrame, 
    collection_name: str, 
    chroma_client: chromadb.ClientAPI, 
    embed_fn: Any
) -> chromadb.Collection:
    """Index documents into ChromaDB."""
    
    # Get or create collection
    collection = chroma_client.get_or_create_collection(
        name=collection_name, 
        embedding_function=embed_fn
    )
    
    # Extract data
    # Note: df MUST have 'processed_context' column (output of VnCoreNLP, no stopword removal)
    if 'processed_context' not in df.columns:
        raise ValueError("DataFrame must contain 'processed_context'.")
        
    documents = df['processed_context'].tolist()
    ids = df.index.astype(str).tolist()
    metadatas = df[['article', 'document']].to_dict(orient='records')
    
    # Add to DB
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    
    return collection
