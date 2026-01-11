# 🔍 Hybrid Search Demo: Semantic + BM25 + RRF

A minimal, single-file Python implementation of a **hybrid search system** that combines:

- **Semantic search** using dense vector embeddings  
- **Keyword search** using BM25  
- **Rank fusion** using **Reciprocal Rank Fusion (RRF)**  

This repository is intended as a **clear, educational reference** for modern retrieval pipelines used in RAG systems, enterprise search, and LLM routing.

---

## ✨ Features

- Sentence-level **semantic similarity search**
- Classical **BM25 keyword matching**
- **Reciprocal Rank Fusion (RRF)** for stable hybrid ranking
- Optional weighting between semantic and lexical signals
- Simple, readable implementation (single Python file)
- No external database required (in-memory corpus)

---

## 🗂 Project Structure

```
.
├── hybrid_search_demo.py
└── README.md
```

---

## ⚙️ Requirements

- Python **3.9+**
- Internet access (for first-time embedding model download)

### Dependencies

```
sentence-transformers
rank-bm25
numpy
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/hybrid-search-demo.git
cd hybrid-search-demo
```

### 2. (Recommended) Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate    # macOS / Linux
# venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install sentence-transformers rank-bm25 numpy
```

> The embedding model will be downloaded automatically on first run.

---

## ▶️ Usage

Basic usage:

```bash
python hybrid_search_demo.py --query "refund policy for subscription"
```

With custom parameters:

```bash
python hybrid_search_demo.py \
  --query "how do I cancel my subscription and get a refund" \
  --topk 5 \
  --rrf_k 60 \
  --semantic_weight 1.0 \
  --bm25_weight 1.0
```

---

## 🧾 Example Output

```
=== Semantic (cosine) Top ===
[0] score=0.7421 | Our refund policy allows returns within 14 days for unused subscriptions.
[3] score=0.6984 | To request a refund, contact support with your invoice number and reason.

=== BM25 Top ===
[0] score=2.1134 | Our refund policy allows returns within 14 days for unused subscriptions.
[3] score=1.8942 | To request a refund, contact support with your invoice number and reason.

=== RRF Fused Top ===
[0] score=0.0323 | Our refund policy allows returns within 14 days for unused subscriptions.
[3] score=0.0315 | To request a refund, contact support with your invoice number and reason.
```

---

## 🔧 How It Works

### 1. Semantic Search
- Uses **Sentence-Transformers** embeddings
- Computes cosine similarity between query and documents

### 2. BM25 Keyword Search
- Uses **rank-bm25**
- Token-based lexical scoring

### 3. Reciprocal Rank Fusion (RRF)

RRF score for a document:

```
RRF(d) = Σ ( w_i / (k + rank_i(d)) )
```

Where:
- `k` controls rank smoothing (commonly `60`)
- `wᵢ` is the weight of each retrieval method

---

## 🧪 Use Cases

- Retrieval-Augmented Generation (RAG)
- Enterprise knowledge search
- Hybrid vector + keyword retrieval
- LLM query routing
- Search research and benchmarking

---

## 🔄 Possible Extensions

- Replace in-memory documents with **Qdrant**, **FAISS**, or **Chroma**
- Add **Relative Score Fusion (RSF)** for comparison
- Expose as a **FastAPI** service
- Add cross-encoder re-ranking
- Integrate directly with LLM response generation

---

## 📚 References

- Cormack et al., *Reciprocal Rank Fusion*, SIGIR 2009  
- Robertson et al., *BM25 and Probabilistic Retrieval*  
- Sentence-Transformers: https://www.sbert.net  

---

## 📜 License

MIT License

---

## ⭐ Notes

This project is intentionally kept small and readable to make hybrid retrieval concepts easy to understand and extend.
