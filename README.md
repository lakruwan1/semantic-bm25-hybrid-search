# 🔍 semantic-bm25-hybrid-search

A hybrid information retrieval combining **semantic vector search** and **BM25 keyword search**, fused using **Reciprocal Rank Fusion (RRF)**.  
This repository provides a clear reference implementation for modern retrieval pipelines used in RAG systems, enterprise search, and LLM routing.

---

## ✨ Features

- Semantic search using dense embeddings
- BM25 keyword-based retrieval
- Reciprocal Rank Fusion (RRF) for hybrid ranking
- Weighted fusion between semantic and lexical signals
- Simple, readable, single-file Python implementation
- No external database required (in-memory corpus)

---

## 🗂 Project Structure

```
.
├── hybrid_search_demo.py
├── images/
│   └── semantic-bm25-hybrid-search.jpeg
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
git clone https://github.com/lakruwan1/semantic-bm25-hybrid-search
cd semantic-bm25-hybrid-search
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

## 🧠 How Hybrid Search Works

<p align="center">
  <img src="images/semantic-bm25-hybrid-search.jpeg" alt="Hybrid Search Flow with Semantic Search, BM25, and RRF" width="700">
</p>

The system retrieves documents using two independent strategies:

1. **Semantic Search** – captures meaning using vector embeddings and cosine similarity  
2. **BM25 Keyword Search** – captures exact lexical relevance  

Each method produces a ranked list, which is fused using **Reciprocal Rank Fusion (RRF)**.

**Reciprocal Rank Fusion (RRF):**

```
RRF(d) = Σᵢ wᵢ / (k + rankᵢ(d))
```

where:
- `k` = RRF constant (commonly 60)
- `wᵢ` = weight of ranking list *i*
- `rankᵢ(d)` = rank position of document *d* in list *i*

---

## 🧪 Use Cases

- Retrieval-Augmented Generation (RAG)
- Enterprise knowledge search
- Hybrid vector + keyword retrieval
- LLM query routing
- Search research and benchmarking

---

## 📚 References

- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* Proceedings of the 32nd ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’09), ACM, pp. 758–759.

- Robertson, S. E., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends® in Information Retrieval, 3(4), 333–389.

- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* Proceedings of EMNLP-IJCNLP 2019. https://www.sbert.net


