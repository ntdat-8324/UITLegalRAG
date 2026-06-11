from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import chromadb

from src.config.settings import settings
from src.preprocessing.vncore_segmenter import VNPreprocessor
from src.preprocessing.stopword_remover import StopwordRemover
from src.retrieval.embedding_function import SbertEmbeddingFunction
from src.retrieval.chromadb_retriever import ChromaDBRetriever
from src.indexing.bm25_indexer import load_bm25_index
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.classification.svm_classifier import SVMDocumentClassifier
from src.classification.predictor import DocumentPredictor
from src.generation.llm_generator import LLMAnswerGenerator
from api.dependencies import app_state
from api.routers import health, query

import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Preprocessors
    print("Loading preprocessors...")
    preprocessor = VNPreprocessor(settings.vncorenlp_dir)
    stopword_remover = StopwordRemover(settings.stopwords_path)
    
    # Check if data files and models exist before fully loading
    # ChromaDB
    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    embed_fn = SbertEmbeddingFunction(model_name=settings.sbert_model_name)
    try:
        collection = chroma_client.get_collection(name="legal_docs", embedding_function=embed_fn)
        app_state.chroma_collection = collection
        chroma_retriever = ChromaDBRetriever(collection, preprocessor)
    except (ValueError, chromadb.errors.InvalidCollectionException):
        print("Warning: ChromaDB collection 'legal_docs' not found. You need to run indexing first.")
        collection = None
        chroma_retriever = None

    # BM25
    bm25_path = os.path.join(settings.model_dir, 'bm25_index.pkl')
    corpus_path = os.path.join(settings.model_dir, 'processed_corpus.pkl')
    bm25_retriever = None
    if os.path.exists(bm25_path) and os.path.exists(corpus_path):
        print("Loading BM25 index and corpus...")
        bm25_model = load_bm25_index(bm25_path)
        import pandas as pd
        df_corpus = pd.read_pickle(corpus_path)
        
        bm25_retriever = BM25Retriever(
            bm25_model=bm25_model,
            documents=df_corpus['processed_context_bm25'].tolist(),
            ids=df_corpus.index.astype(str).tolist(),
            stopword_remover=stopword_remover,
            preprocessor=preprocessor,
            df=df_corpus,
            limit=settings.top_k_retrieval
        )
    else:
        print("Warning: BM25 index or corpus not found. Run indexing first.")
        
    # Hybrid Retriever
    if chroma_retriever and bm25_retriever:
        app_state.hybrid_retriever = HybridRetriever(
            biencoder=chroma_retriever,
            bm25=bm25_retriever,
            topk=settings.top_k_retrieval,
            alpha=settings.hybrid_alpha
        )
        
    # Cross-Encoder Reranker
    print(f"Loading Reranker ({settings.cross_encoder_model_name})...")
    app_state.reranker = CrossEncoderReranker(model_name=settings.cross_encoder_model_name, preprocessor=preprocessor)
    
    # Document Classifier
    print("Loading Document Classifier...")
    try:
        classifier = SVMDocumentClassifier()
        classifier.load(settings.model_dir)
        app_state.predictor = DocumentPredictor(
            model=classifier.model,
            vectorizer=classifier.vectorizer,
            label_encoder=classifier.label_encoder,
            preprocessor=preprocessor,
            stopword_remover=stopword_remover
        )
    except FileNotFoundError:
        print("Warning: Classifier models not found. Run train_svm.py first.")
        
    # LLM Generator
    print("Initializing LLM generator...")
    app_state.llm_generator = LLMAnswerGenerator(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model
    )
    
    print("Startup complete.")
    yield
    print("Shutting down...")

app = FastAPI(title="Vietnamese Legal RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(query.router, prefix="/api", tags=["Query"])
