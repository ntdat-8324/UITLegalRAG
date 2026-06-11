import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from src.config.settings import settings
from src.indexing.data_loader import load_and_preprocess_data
from src.preprocessing.vncore_segmenter import VNPreprocessor
from src.preprocessing.stopword_remover import StopwordRemover
from src.retrieval.embedding_function import SbertEmbeddingFunction
from src.indexing.chroma_indexer import build_chroma_index
from src.indexing.bm25_indexer import build_bm25_index, save_bm25_index

def main():
    print("Starting indexing pipeline...")
    
    train_path = os.path.join(settings.data_dir, 'train.csv')
    if not os.path.exists(train_path):
        print(f"Error: Data not found at {train_path}")
        return
        
    print("1. Loading and cleaning data...")
    df = load_and_preprocess_data(train_path)
    if df.empty:
        return
    print(f"Loaded {len(df)} unique context records.")
    
    print("2. Initializing NLP tools...")
    preprocessor = VNPreprocessor(settings.vncorenlp_dir)
    stopword_remover = StopwordRemover(settings.stopwords_path)
    
    print("3. Preprocessing text for ChromaDB (word segmenting only)...")
    df['processed_context'] = df['context'].apply(preprocessor.preprocess)
    
    print("4. Preprocessing text for BM25 (removing stopwords)...")
    df['processed_context_bm25'] = df['processed_context'].apply(stopword_remover.remove_stopwords)
    
    print("5. Building ChromaDB Index...")
    embed_fn = SbertEmbeddingFunction(model_name=settings.sbert_model_name)
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = build_chroma_index(df, "legal_docs", chroma_client, embed_fn)
    print(f"ChromaDB collection created with {collection.count()} documents.")
    
    print("6. Building BM25 Index...")
    bm25_model = build_bm25_index(df, k1=settings.bm25_k1, b=settings.bm25_b)
    
    # Save BM25 to models directory
    os.makedirs(settings.model_dir, exist_ok=True)
    bm25_path = os.path.join(settings.model_dir, 'bm25_index.pkl')
    save_bm25_index(bm25_model, bm25_path)
    
    # Also save the df so we can load it for ids and documents later
    df_path = os.path.join(settings.model_dir, 'processed_corpus.pkl')
    df.to_pickle(df_path)
    print(f"BM25 index and corpus saved to {settings.model_dir}")
    
    print("Indexing complete!")

if __name__ == "__main__":
    main()
