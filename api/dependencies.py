from typing import Optional
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.classification.predictor import DocumentPredictor
from src.generation.llm_generator import LLMAnswerGenerator
import chromadb

# Global instances (to be initialized on startup)
class AppState:
    hybrid_retriever: Optional[HybridRetriever] = None
    reranker: Optional[CrossEncoderReranker] = None
    predictor: Optional[DocumentPredictor] = None
    llm_generator: Optional[LLMAnswerGenerator] = None
    chroma_collection: Optional[chromadb.Collection] = None

app_state = AppState()

def get_hybrid_retriever() -> HybridRetriever:
    return app_state.hybrid_retriever

def get_reranker() -> CrossEncoderReranker:
    return app_state.reranker

def get_predictor() -> DocumentPredictor:
    return app_state.predictor

def get_llm_generator() -> LLMAnswerGenerator:
    return app_state.llm_generator

def get_chroma_collection() -> chromadb.Collection:
    return app_state.chroma_collection
