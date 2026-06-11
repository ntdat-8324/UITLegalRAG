# Vietnamese Legal Hybrid RAG System

An end-to-end Hybrid RAG (Retrieval-Augmented Generation) system specialized for Vietnamese legal documents. This project implements a high-performance retrieval pipeline combining BM25 (sparse retrieval), SBERT (dense retrieval), Cross-Encoder reranking, SVM classification, and an LLM generator to ensure highly accurate and context-aware legal consultation.

## System Pipeline Architecture

The system answers questions through a sophisticated multi-stage pipeline to ensure maximum accuracy:

1. **Document Classification (SVM Filter)**: When a question is asked, an SVM classifier predicts which legal document(s) the question belongs to (e.g., "Quy dinh dao tao", "Luat giao thong"). This acts as a hard filter to drastically reduce the search space.
2. **Hybrid Retrieval (BM25 + SBERT)**: 
   - **BM25** performs keyword-based lexical search.
   - **SBERT** (`bkai-foundation-models/vietnamese-bi-encoder`) performs dense semantic search.
   - Both results are combined. The system retrieves a large pool of candidates (e.g., Top 15) to ensure high recall.
3. **Cross-Encoder Reranking (PhoRanker)**: The top candidates are passed to `itdainb/PhoRanker`. The Reranker evaluates the logical relationship between the question and each context, re-scoring and sorting them to pick the absolute most relevant top 5 documents.
4. **LLM Generation with Streaming**: The top 5 contexts are injected into a prompt and sent to Gemma 2. The response is streamed back to the client using Server-Sent Events (SSE) for a real-time typing effect.

## Key Features

- **Hybrid Retrieval**: Effectively fuses Sparse Retrieval (BM25Plus) with Dense Retrieval.
- **Cross-Encoder Reranking**: Re-evaluates and scores retrieved documents using `itdainb/PhoRanker` to maximize relevance.
- **Document Classification**: Support Vector Machine (SVM) combined with TF-IDF filtering.
- **LLM Answer Generation**: Leverages the `Gemma 2 9B IT` model via OpenRouter API.
- **Real-time Streaming (SSE)**: Provides a ChatGPT-like typing effect.
- **VnCoreNLP Preprocessing**: Robust and accurate Vietnamese word segmentation.
- **Modern UI**: A premium, responsive chatbot interface built with React and Vite.

## Tech Stack

- **Backend**: FastAPI, Python, Uvicorn
- **Frontend**: React 18, Vite, Lucide React
- **Vector Database**: ChromaDB
- **Machine Learning**: Scikit-Learn (SVM), Sentence-Transformers, Rank-BM25, HuggingFace
- **LLM Provider**: OpenRouter (`google/gemma-2-9b-it:free`)

---

## Project Structure

```
.
├── api/                  # FastAPI application and endpoints
├── frontend/             # React JS user interface
├── scripts/              # Setup, training, and indexing scripts
├── src/                  # Core system logic
│   ├── classification/   # SVM document classification pipeline
│   ├── config/           # Pydantic settings and configurations
│   ├── generation/       # LLM API integration
│   ├── indexing/         # ChromaDB and BM25 index builders
│   ├── preprocessing/    # VnCoreNLP segmenter and Stopwords remover
│   └── retrieval/        # Hybrid Retriever and Reranker
├── .env.example          # Example environment variables
└── requirements.txt      # Python dependencies
```

---

## Installation & Setup Guide

### 1. Prerequisites
- Python 3.11+
- Java JDK 8+ (Strictly required for VnCoreNLP)
- Node.js & npm (For the frontend)
- uv - An extremely fast Python package installer (`pip install uv`)

### 2. Backend Setup

Create and activate a virtual environment using `uv`, then install dependencies:

```bash
uv venv myvenv
# On Windows:
myvenv\Scripts\activate
uv pip install -r requirements.txt
```

Configure environment variables:
- Copy `.env.example` to `.env`
- Open `.env` and set your OpenRouter API Key.

### 3. Data Preparation

1. Create a `data/raw/` directory and place your `train.csv` inside. The CSV must contain `question`, `context`, `document`, and `article` columns.
2. Place your Vietnamese stopwords file at `data/stopwords/vietnamese-stopwords.txt`.

### 4. Setup NLP Pipelines & Indexing

Execute the following scripts in order:

**Step 4.1: Install VnCoreNLP**
```bash
python scripts/setup_vncorenlp.py
```

**Step 4.2: Train the SVM Classifier**
```bash
python scripts/train_svm.py
```

**Step 4.3: Build Vector DB & BM25 Index**
```bash
python scripts/index_data.py
```

### 5. Frontend Setup

```bash
cd frontend
npm install
```

---

## Running the Application

### Start the FastAPI Backend
```bash
uvicorn api.main:app --reload
```
*The API will be available at http://localhost:8000*

### Start the React Frontend
```bash
cd frontend
npm run dev
```

---

## Troubleshooting

1. **VnCoreNLP/Java Errors**: Ensure Java JDK is installed and the `JAVA_HOME` environment variable is correctly set and added to your system PATH.
2. **Missing Authentication header (LLM)**: Verify that your `OPENROUTER_API_KEY` in the `.env` file is valid and contains no trailing spaces.
3. **InvalidCollectionException**: This occurs if the Backend is started before `scripts/index_data.py` finishes successfully.
