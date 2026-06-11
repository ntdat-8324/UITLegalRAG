# Vietnamese Legal Hybrid RAG System

An end-to-end Hybrid RAG (Retrieval-Augmented Generation) system specialized for Vietnamese legal documents. This project implements a high-performance retrieval pipeline combining BM25 (sparse retrieval), SBERT (dense retrieval), Cross-Encoder reranking, SVM classification, and an LLM generator to ensure highly accurate and context-aware legal consultation.

## ✨ Key Features

- **Hybrid Retrieval**: Effectively fuses Sparse Retrieval (BM25Plus) with Dense Retrieval (SBERT using `bkai-foundation-models/vietnamese-bi-encoder`).
- **Cross-Encoder Reranking**: Re-evaluates and scores retrieved documents using `itdainb/PhoRanker` to maximize relevance.
- **Document Classification**: Utilizes a Support Vector Machine (SVM) combined with TF-IDF to filter and narrow down the search space, significantly boosting both retrieval speed and accuracy.
- **LLM Answer Generation**: Leverages the `Gemma 2 9B IT` model via OpenRouter API, instructed with carefully crafted prompts tailored for Vietnamese legal consulting.
- **VnCoreNLP Preprocessing**: Ensures robust and accurate Vietnamese word segmentation.
- **Modern UI**: A premium, responsive chatbot interface built with React and Vite, featuring dark mode, chat history, and bookmarking capabilities.

## 🛠 Tech Stack

- **Backend**: FastAPI, Python, Uvicorn
- **Frontend**: React 18, Vite, Lucide React
- **Vector Database**: ChromaDB
- **Machine Learning**: Scikit-Learn (SVM), Sentence-Transformers, Rank-BM25, HuggingFace
- **LLM Provider**: OpenRouter (`google/gemma-2-9b-it:free`)

---

## 📂 Project Structure

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

## 🚀 Installation & Setup Guide

### 1. Prerequisites
- [Python 3.11+](https://www.python.org/)
- [Java JDK 8+](https://www.oracle.com/java/technologies/downloads/) (Strictly required for VnCoreNLP)
- [Node.js & npm](https://nodejs.org/) (For the frontend)
- [uv](https://github.com/astral-sh/uv) - An extremely fast Python package installer (`pip install uv`)

### 2. Backend Setup

Create and activate a virtual environment using `uv`, then install dependencies:

```bash
# Create a virtual environment named 'myvenv'
uv venv myvenv

# Activate the virtual environment
# On Windows:
myvenv\Scripts\activate
# On Linux/macOS:
# source myvenv/bin/activate

# Install dependencies blazing fast
uv pip install -r requirements.txt
```

Configure environment variables:
- Copy `.env.example` to `.env`
- Open `.env` and set your OpenRouter API Key in the `OPENROUTER_API_KEY` variable.

### 3. Data Preparation

The system expects raw data files to be placed in `data/raw/` and stopwords in `data/stopwords/`:
1. Create a `data/raw/` directory and place your `train.csv` (and optionally `val.csv`, `test.csv`) inside. The CSV must contain `question`, `context`, `document`, and `article` columns.
2. Place your Vietnamese stopwords file at `data/stopwords/vietnamese-stopwords.txt`.

### 4. Setup NLP Pipelines & Indexing

Ensure your virtual environment is activated and execute the following scripts in order from the root directory:

**Step 4.1: Install VnCoreNLP**
Downloads the required Vietnamese word segmentation models to your local machine (`~/.vncorenlp`).
```bash
python scripts/setup_vncorenlp.py
```

**Step 4.2: Train the SVM Classifier**
Trains the ML classifier used to filter relevant legal documents.
```bash
python scripts/train_svm.py
```

**Step 4.3: Build Vector DB & BM25 Index**
Embeds the legal corpus and builds indices. *(This step may take a few minutes as SBERT processes the vectors).*
```bash
python scripts/index_data.py
```

### 5. Frontend Setup

Open a new Terminal/Command Prompt window:

```bash
cd frontend
npm install
```

---

## 🏃 Running the Application

### Start the FastAPI Backend
From the project root directory (ensure the virtual environment is activated):
```bash
uvicorn api.main:app --reload
```
*The API will be available at http://localhost:8000*
*Swagger UI Documentation: http://localhost:8000/docs*

### Start the React Frontend
In your frontend terminal (`/frontend` directory):
```bash
npm run dev
```
*The web application will be accessible at the displayed local URL (typically http://localhost:5173)*

---

## 💡 Troubleshooting

1. **VnCoreNLP/Java Errors**: Ensure Java JDK is installed and the `JAVA_HOME` environment variable is correctly set and added to your system PATH.
2. **Missing Authentication header (LLM)**: Verify that your `OPENROUTER_API_KEY` in the `.env` file is valid and contains no trailing spaces.
3. **InvalidCollectionException**: This occurs if the Backend is started before `scripts/index_data.py` finishes successfully. Ensure indexing is 100% complete before starting Uvicorn.
