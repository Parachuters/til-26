# NLP Solution Plan — Advanced Track

## Problem Summary

Answer questions using **Retrieval-Augmented Generation (RAG)** on a provided document corpus set in a fictional cyberpunk world (Clairos). Advanced track includes question levels L1–L5.

**Scoring:**
- Retrieval: Top-3 returned doc IDs must overlap with ground truth
- Correctness: ModernBERT-base answer equivalence model (threshold 0.9)
- Partial credit: 0.4 if retrieval correct but answer wrong
- L4 (unanswerable, no reference): return `{"documents": [], "answer": ""}`
- L5 (unanswerable, false premise): return `{"documents": [], "answer": ""}`

**Interface:** POST `/nlp` port `5004`
- First request: corpus loading (return `{"predictions": [{"status": "loaded"}]}`)
- Subsequent requests: QA (return `documents` + `answer` per question)

---

## Question Level Definitions

| Level | Description | Strategy |
|---|---|---|
| L1 | Direct extraction from single doc | Standard RAG |
| L2 | Single inference step from one doc | RAG + LLM reasoning |
| L3 | Cross-document synthesis | RAG with multi-doc context |
| L4 | Unanswerable, no relevant docs exist | Return empty |
| L5 | Unanswerable due to false premise | Return empty |

---

## Recommended Architecture

```
Question → [Embedding Model] → Vector Search → [Top-K Docs]
         → [LLM with RAG Prompt] → Answer + Unanswerable Detection
```

### Component Choices

| Component | Recommended | Why |
|---|---|---|
| Embedding | `BAAI/bge-m3` or `all-mpnet-base-v2` | High retrieval quality, multilingual support |
| Vector Store | FAISS (flat L2 or IVF) | Fast, no external service needed |
| LLM | Qwen2.5-7B-Instruct or Mistral-7B-Instruct | Fits in GPU, good reasoning |
| Reranker | `BAAI/bge-reranker-v2-m3` | Improves top-K precision (optional) |

---

## Implementation Plan

### 1. Corpus Loading (`load_corpus`)

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class NLPManager:
    def __init__(self):
        self.embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
        self.llm = load_llm()  # see section 3
        self.index = None
        self.doc_ids = []
        self.doc_texts = []

    def load_corpus(self, documents: list[dict[str, str]]) -> None:
        texts = [d["document"] for d in documents]
        ids = [d["id"] for d in documents]
        
        # Chunk long documents (>512 tokens) for better retrieval
        chunks, chunk_ids = chunk_documents(texts, ids, max_tokens=400, overlap=50)
        
        embeddings = self.embedder.encode(
            chunks, batch_size=64, normalize_embeddings=True, show_progress_bar=True
        )
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine sim (with normalized vecs)
        self.index.add(embeddings.astype(np.float32))
        self.doc_ids = chunk_ids
        self.doc_texts = chunks
        self.loaded = True
```

### 2. Chunking Strategy

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

def chunk_documents(texts, ids, max_tokens=400, overlap=50):
    chunks, chunk_ids = [], []
    for doc_id, text in zip(ids, texts):
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            chunks.append(text)
            chunk_ids.append(doc_id)
        else:
            for start in range(0, len(tokens), max_tokens - overlap):
                chunk_tokens = tokens[start:start + max_tokens]
                chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
                chunk_ids.append(doc_id)
    return chunks, chunk_ids
```

### 3. LLM Setup

**Option A: Local Qwen2.5-7B-Instruct (recommended for self-contained container)**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_llm():
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda"
    )
    return tokenizer, model
```

**Option B: Claude API (via Anthropic SDK) — if internet access is allowed**

```python
import anthropic
client = anthropic.Anthropic()
```

Use Option A for air-gapped competition environment. Pre-download weights to Docker image.

### 4. QA Pipeline (`qa`)

```python
def qa(self, question: str) -> dict:
    # Step 1: Embed question
    q_emb = self.embedder.encode([question], normalize_embeddings=True)
    
    # Step 2: Retrieve top-K chunks
    K = 5  # retrieve more, then deduplicate
    scores, indices = self.index.search(q_emb.astype(np.float32), K)
    
    retrieved_ids = [self.doc_ids[i] for i in indices[0]]
    retrieved_texts = [self.doc_texts[i] for i in indices[0]]
    
    # Deduplicate doc IDs (keep order, max 3 unique)
    seen, top3_ids, top3_texts = set(), [], []
    for doc_id, text in zip(retrieved_ids, retrieved_texts):
        if doc_id not in seen:
            seen.add(doc_id)
            top3_ids.append(doc_id)
            top3_texts.append(text)
        if len(top3_ids) == 3:
            break
    
    # Step 3: Check unanswerable (L4/L5)
    max_score = float(scores[0][0])
    if max_score < UNANSWERABLE_THRESHOLD:  # tune this (try 0.4-0.6)
        return {"documents": [], "answer": ""}
    
    # Step 4: Generate answer with LLM
    context = "\n\n".join(f"[{doc_id}]: {text}" for doc_id, text in zip(top3_ids, top3_texts))
    answer = generate_answer(question, context)
    
    # Step 5: Post-check for unanswerable based on LLM output
    if is_unanswerable(answer):
        return {"documents": [], "answer": ""}
    
    return {"documents": top3_ids, "answer": answer}
```

### 5. Unanswerable Detection (L4/L5)

Two-pronged approach:

**A. Retrieval score threshold:**
```python
UNANSWERABLE_THRESHOLD = 0.5  # cosine similarity; tune on validation set
if max_score < UNANSWERABLE_THRESHOLD:
    return {"documents": [], "answer": ""}
```

**B. LLM prompt for unanswerable:**
```python
SYSTEM_PROMPT = """You are a QA assistant. Answer questions using ONLY the provided context.
If the context does not contain enough information to answer, or if the question contains 
a false premise, respond with exactly: UNANSWERABLE"""

def generate_answer(question, context):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    output = run_llm(SYSTEM_PROMPT, prompt)
    return output.strip()

def is_unanswerable(answer: str) -> bool:
    return "UNANSWERABLE" in answer.upper()
```

### 6. Cross-Document Questions (L3)

Ensure top-K retrieval fetches chunks from multiple documents by:
- Retrieving K=8 candidates, then deduplicating to top-3 unique doc IDs
- Including all 3 context texts in the LLM prompt

---

## Prompt Engineering

```python
SYSTEM_PROMPT = """You are a precise QA assistant operating in the Clairos cyberpunk world.
Rules:
1. Answer using ONLY the provided document excerpts.
2. Be concise — one sentence unless more is needed.
3. If the question has a false premise or no answer exists in the documents, say: UNANSWERABLE
4. Do not hallucinate facts not in the context."""

def build_prompt(question: str, context: str) -> str:
    return f"""Documents:
{context}

Question: {question}
Answer (or UNANSWERABLE):"""
```

---

## Optional: Reranking for Better Retrieval

After initial FAISS retrieval, use a cross-encoder reranker for better precision:

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")

pairs = [(question, text) for text in retrieved_texts]
rerank_scores = reranker.predict(pairs)
sorted_idx = np.argsort(rerank_scores)[::-1]
top3_ids = [retrieved_ids[i] for i in sorted_idx[:3]]
```

Trade-off: adds ~100ms per query; worth it for L3/L4/L5 precision.

---

## Dockerfile Notes

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
RUN pip install sentence-transformers faiss-gpu transformers accelerate
# Pre-download models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

## requirements.txt

```
sentence-transformers>=2.7.0
faiss-gpu  # or faiss-cpu if no GPU
transformers>=4.40.0
accelerate
torch>=2.1.0
```

---

## Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| L4/L5 false positives (real question called unanswerable) | Tune threshold carefully; use both retrieval score + LLM check |
| LLM hallucination | Use strict system prompt; temperature=0 for determinism |
| Corpus loading too slow | Use batch encoding; run in background thread (already in server) |
| LLM too slow for full test set | Use quantized model (4-bit AWQ/GPTQ); or smaller model |
| Retrieval fails for L3 cross-doc | Retrieve K=8-10, deduplicate to top-3 doc IDs |

---

## Scoring Checklist

- [ ] L4/L5: return `{"documents": [], "answer": ""}` exactly
- [ ] Only first 3 doc IDs are scored; return max 3
- [ ] Test unanswerable threshold on all 5 question levels
- [ ] Verify corpus loading returns `{"predictions": [{"status": "loaded"}]}`
- [ ] Measure end-to-end QA latency; stay within 30-min budget
- [ ] Confirm answer equivalence style matches ModernBERT-base threshold 0.9
