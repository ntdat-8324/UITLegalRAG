from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
import json
from api.schemas.request import QueryRequest
from api.schemas.response import QueryResponse, SourceDocument
from api.dependencies import get_hybrid_retriever, get_reranker, get_predictor, get_llm_generator, get_chroma_collection
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.classification.predictor import DocumentPredictor
from src.generation.llm_generator import LLMAnswerGenerator
import time

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_legal(
    req: QueryRequest,
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    reranker: CrossEncoderReranker = Depends(get_reranker),
    predictor: DocumentPredictor = Depends(get_predictor),
    llm: LLMAnswerGenerator = Depends(get_llm_generator),
    collection = Depends(get_chroma_collection)
):
    start_time = time.time()
    
    # 1. Classification (optional)
    doc_filters = None
    if req.use_classifier and predictor and hasattr(predictor, 'model'):
        # Just safely predict if model is loaded
        try:
            doc_filters = predictor.predict_top_k(req.question, k=15)
        except Exception as e:
            print(f"Classifier error: {e}")
            doc_filters = None
            
    # 2. Hybrid Retrieval
    top_doc_ids = []
    if retriever:
        # Increase retrieve count if using reranker
        retrieve_limit = req.top_k * 3 if req.use_reranker else req.top_k
        top_doc_ids = retriever.query(req.question, document_filters=doc_filters, limit=retrieve_limit)
        
    # Gather document data from ChromaDB
    candidates = []
    if top_doc_ids and collection:
        results = collection.get(ids=top_doc_ids)
        for i, doc_id in enumerate(results['ids']):
            candidates.append({
                'doc_id': doc_id,
                'context': results['documents'][i],
                'article': results['metadatas'][i].get('article', ''),
                'document': results['metadatas'][i].get('document', '')
            })
            
    # 3. Cross-Encoder Reranking (optional)
    if req.use_reranker and reranker and candidates:
        candidates = reranker.rerank(req.question, candidates, top_k=req.top_k)
    else:
        # Just take top_k if no reranking
        candidates = candidates[:req.top_k]
        
    # 4. LLM Generation
    contexts = [c['context'] for c in candidates]
    answer = ""
    if llm:
        answer = llm.generate_answer(req.question, contexts)
        
    # Build response
    sources = []
    for i, c in enumerate(candidates):
        sources.append(SourceDocument(
            doc_id=c['doc_id'],
            context=c['context'],
            article=c['article'],
            document=c['document'],
            score=1.0 - (i*0.01), # Dummy descending score if not mapped
            rerank_score=c.get('rerank_score')
        ))
        
    end_time = time.time()
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        processing_time_ms=(end_time - start_time) * 1000
    )

@router.post("/query_stream")
async def query_legal_stream(
    req: QueryRequest,
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    reranker: CrossEncoderReranker = Depends(get_reranker),
    predictor: DocumentPredictor = Depends(get_predictor),
    llm: LLMAnswerGenerator = Depends(get_llm_generator),
    collection = Depends(get_chroma_collection)
):
    async def generate_events():
        # 1. Classification (optional)
        doc_filters = None
        if req.use_classifier and predictor and hasattr(predictor, 'model'):
            try:
                doc_filters = predictor.predict_top_k(req.question, k=15)
            except Exception as e:
                print(f"Classifier error: {e}")
                doc_filters = None
                
        # 2. Hybrid Retrieval
        top_doc_ids = []
        if retriever:
            retrieve_limit = req.top_k * 3 if req.use_reranker else req.top_k
            top_doc_ids = retriever.query(req.question, document_filters=doc_filters, limit=retrieve_limit)
            
        candidates = []
        if top_doc_ids and collection:
            results = collection.get(ids=top_doc_ids)
            for i, doc_id in enumerate(results['ids']):
                candidates.append({
                    'doc_id': doc_id,
                    'context': results['documents'][i],
                    'article': results['metadatas'][i].get('article', ''),
                    'document': results['metadatas'][i].get('document', '')
                })
                
        # 3. Cross-Encoder Reranking
        if req.use_reranker and reranker and candidates:
            candidates = reranker.rerank(req.question, candidates, top_k=req.top_k)
        else:
            candidates = candidates[:req.top_k]
            
        # Build sources list
        sources = []
        for i, c in enumerate(candidates):
            sources.append(SourceDocument(
                doc_id=c['doc_id'],
                context=c['context'],
                article=c['article'],
                document=c['document'],
                score=1.0 - (i*0.01),
                rerank_score=c.get('rerank_score')
            ).dict())
            
        # Yield sources as first event
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
        
        # 4. LLM Generation Streaming
        contexts = [c['context'] for c in candidates]
        if llm:
            for chunk in llm.generate_answer_stream(req.question, contexts):
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                
        # Yield done event
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate_events(), media_type="text/event-stream")
