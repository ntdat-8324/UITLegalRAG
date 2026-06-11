# Vietnamese Legal Hybrid RAG System

Hệ thống Hybrid RAG (Retrieval-Augmented Generation) chuyên biệt cho văn bản pháp luật Việt Nam (Quy chế, Điều luật đại học và các văn bản quy phạm pháp luật khác). Hệ thống kết hợp giữa tìm kiếm từ khóa (BM25), tìm kiếm ngữ nghĩa (SBERT), xếp hạng lại (Cross-encoder PhoRanker) và sinh câu trả lời bằng LLM (Gemma 2) để đảm bảo độ chính xác cao nhất cho ngữ cảnh pháp lý.

## ✨ Tính năng chính

- **Hybrid Retrieval**: Kết hợp linh hoạt giữa Sparse Retrieval (BM25Plus) và Dense Retrieval (SBERT `bkai-foundation-models/vietnamese-bi-encoder`).
- **Reranking**: Sử dụng Cross-encoder (`itdainb/PhoRanker`) để chấm điểm lại mức độ liên quan của các tài liệu tìm được.
- **Classification**: Tích hợp mô hình học máy SVM (Support Vector Machine) với TF-IDF để phân loại và giới hạn không gian tìm kiếm, giúp tăng tốc độ và độ chính xác.
- **LLM Answer Generation**: Sử dụng mô hình `Gemma 2 9B IT` thông qua OpenRouter API, được prompt thiết kế riêng cho tư vấn pháp luật.
- **VnCoreNLP Preprocessing**: Đảm bảo chất lượng tách từ (word segmentation) tiếng Việt.
- **Modern UI**: Giao diện Chatbot chuyên nghiệp xây dựng bằng React/Vite với Dark mode, History và Bookmarks.

## 🛠 Tech Stack

- **Backend**: FastAPI, Python, Uvicorn
- **Frontend**: React 18, Vite, Lucide React
- **Vector Database**: ChromaDB
- **Machine Learning**: Scikit-Learn (SVM), Sentence-Transformers, Rank-BM25, HuggingFace
- **LLM API**: OpenRouter (`google/gemma-2-9b-it:free`)

---

## 📂 Cấu trúc dự án

```
.
├── api/                  # FastAPI app và endpoints
├── frontend/             # Ứng dụng React JS
├── scripts/              # Các kịch bản cài đặt, huấn luyện và đánh giá
├── src/                  # Mã nguồn lõi (Core logic)
│   ├── classification/   # Pipeline phân loại tài liệu (SVM)
│   ├── config/           # Cấu hình hệ thống (Pydantic settings)
│   ├── generation/       # Gọi API LLM
│   ├── indexing/         # Đánh chỉ mục dữ liệu (BM25, ChromaDB)
│   ├── preprocessing/    # Tiền xử lý (VnCoreNLP, Stopwords)
│   └── retrieval/        # Tìm kiếm và Reranking
├── .env.example          # Mẫu cấu hình biến môi trường
└── requirements.txt      # Thư viện Python cần thiết
```

---

## 🚀 Hướng dẫn cài đặt và chạy hệ thống

### 1. Yêu cầu hệ thống (Prerequisites)
- [Python 3.10+](https://www.python.org/)
- [Java JDK 8+](https://www.oracle.com/java/technologies/downloads/) (Bắt buộc cho VnCoreNLP)
- [Node.js & npm](https://nodejs.org/) (Cho frontend)
- [uv](https://github.com/astral-sh/uv) - Trình quản lý package Python cực nhanh (Cài đặt bằng lệnh `pip install uv` nếu chưa có)

### 2. Thiết lập Backend (Python)

Tạo virtual environment với tên `myvenv` bằng `uv`:

```bash
# Tạo virtual environment
uv venv myvenv

# Kích hoạt virtual environment
# Windows:
myvenv\Scripts\activate
# Linux/macOS:
# source myvenv/bin/activate

# Cài đặt các thư viện cần thiết siêu tốc với uv
uv pip install -r requirements.txt
```

Thiết lập biến môi trường:
- Copy file `.env.example` thành `.env`
- Mở file `.env` và thêm khóa API OpenRouter của bạn vào biến `OPENROUTER_API_KEY`

### 3. Chuẩn bị Dữ liệu (Data Preparation)

Hệ thống cần các file dữ liệu tại thư mục `data/raw/` và `data/stopwords/`:
1. Tạo thư mục `data/raw/` và đặt file `train.csv` (Cùng với val.csv, test.csv nếu cần) vào đó. (File CSV phải có cột `question`, `context`, `document`, `article`).
2. Đặt file stopwords tiếng Việt vào `data/stopwords/vietnamese-stopwords.txt`.

### 4. Setup Pipelines & Indexing

Đảm bảo bạn vẫn đang ở trong `myvenv` và chạy lần lượt các script sau từ thư mục gốc:

**Bước 4.1: Cài đặt VnCoreNLP**
Tải mô hình phân mảnh từ tiếng Việt:
```bash
python scripts/setup_vncorenlp.py
```

**Bước 4.2: Huấn luyện mô hình SVM**
Huấn luyện Classifier để lọc Document liên quan:
```bash
python scripts/train_svm.py
```

**Bước 4.3: Xây dựng Vector DB & BM25 Index**
Nhúng dữ liệu (Embedding) và xây dựng chỉ mục:
*(Quá trình này có thể mất một chút thời gian do phải chạy mô hình SBERT để tạo vector)*
```bash
python scripts/index_data.py
```

### 5. Thiết lập Frontend (React)

Mở một cửa sổ Terminal/Command Prompt mới:

```bash
cd frontend
npm install
```

---

## 🏃 Khởi chạy Hệ thống

### Bật FastAPI Backend
Từ thư mục gốc của dự án (hãy chắc chắn `myvenv` đang được kích hoạt):
```bash
uvicorn api.main:app --reload
```
*API sẽ chạy tại http://localhost:8000*
*Swagger Docs: http://localhost:8000/docs*

### Bật React Frontend
Mở cửa sổ Terminal/Command Prompt dành cho frontend (thư mục `/frontend`):
```bash
npm run dev
```
*Ứng dụng web sẽ chạy tại đường dẫn được hiển thị (Thường là http://localhost:5173)*

---

## 💡 Xử lý lỗi thường gặp

1. **Lỗi VnCoreNLP (Java)**: Đảm bảo Java đã có trong PATH của hệ thống. Bạn có thể kiểm tra bằng lệnh `java -version`.
2. **Lỗi ChromaDB Module**: Nếu ChromaDB báo lỗi, hãy đảm bảo bạn dùng C++ build tools bản mới nhất nếu dùng Windows. `uv` xử lý wheels khá tốt nhưng đôi khi C++ Build tools là bắt buộc.
3. **Frontend gọi API bị lỗi Network**: Kiểm tra biến `API_BASE` trong `frontend/src/App.jsx` xem có trùng khớp với cổng Backend (mặc định 8000) hay không.
