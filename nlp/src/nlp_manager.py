"""Manages the NLP model."""

import os
import re
import threading

import numpy as np

try:
    import faiss
    _faiss_available = True
except ImportError:
    _faiss_available = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    _st_available = True
except ImportError:
    _st_available = False

try:
    from rank_bm25 import BM25Okapi
    _bm25_available = True
except ImportError:
    _bm25_available = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        BitsAndBytesConfig = None
    import torch
    _hf_available = True
except ImportError:
    BitsAndBytesConfig = None
    _hf_available = False

try:
    import bitsandbytes  # noqa: F401 — presence check only
    _bnb_available = True
except ImportError:
    _bnb_available = False


EMBED_MODEL = os.getenv("NLP_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("NLP_RERANKER_MODEL", "BAAI/bge-reranker-base")
LLM_MODEL = os.getenv("NLP_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
UNANSWERABLE_THRESHOLD = float(os.getenv("NLP_UNANSWERABLE_THRESHOLD", "0.35"))
TOP_K = int(os.getenv("NLP_TOP_K", "40"))   # retrieve more candidates before reranking
RERANK_TOP_K = int(os.getenv("NLP_RERANK_TOP_K", "20"))
MAX_NEW_TOKENS = int(os.getenv("NLP_MAX_NEW_TOKENS", "64"))
CHUNK_WORDS = int(os.getenv("NLP_CHUNK_WORDS", "420"))
OVERLAP_WORDS = int(os.getenv("NLP_OVERLAP_WORDS", "80"))
EMBED_BATCH_SIZE = int(os.getenv("NLP_EMBED_BATCH_SIZE", "128"))
# BM25 weight in the hybrid score (1-BM25_WEIGHT goes to dense retrieval).
BM25_WEIGHT = float(os.getenv("NLP_BM25_WEIGHT", "0.3"))
# Use 4-bit quantisation for the LLM when bitsandbytes is available.
USE_4BIT = os.getenv("NLP_USE_4BIT", "1") not in ("0", "false", "False", "no")
BGE_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)
_TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

# System prompt grounded in the Clairos fictional setting so the LLM does not
# confuse in-world terminology with real-world knowledge.
SYSTEM_PROMPT = (
    "You are a precise QA assistant for the world of Clairos — a cyberpunk "
    "setting after the Cascade flood where megacorporations rule Haven, the "
    "last great city. Answer questions using ONLY the provided document "
    "excerpts.\n"
    "Rules:\n"
    "1. Be concise — one sentence unless more is needed.\n"
    "2. If the question has a false premise or no answer exists in the "
    "documents, respond with exactly: UNANSWERABLE\n"
    "3. Do not use outside knowledge or hallucinate facts not in the context."
)


def _tokenize_for_sparse(text: str) -> list[str]:
    """Tokenize text for lexical retrieval without losing named entities."""
    return [
        token.strip("'").lower()
        for token in _TOKEN_RE.findall(text)
        if token.strip("'")
    ]


def _format_embed_query(question: str) -> str:
    """BGE-style query instruction improves asymmetric retrieval quality."""
    if question.startswith(BGE_QUERY_INSTRUCTION):
        return question
    return f"{BGE_QUERY_INSTRUCTION}{question}"


def _chunk_documents(
    texts,
    ids,
    max_words: int = CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
):
    chunks, chunk_ids = [], []
    for doc_id, text in zip(ids, texts):
        matches = list(_TOKEN_RE.finditer(text))
        if len(matches) <= max_words:
            chunks.append(text)
            chunk_ids.append(doc_id)
        else:
            start = 0
            step = max(1, max_words - overlap_words)
            while start < len(matches):
                end = min(start + max_words, len(matches))
                char_start = matches[start].start()
                char_end = matches[end - 1].end()
                chunks.append(text[char_start:char_end].strip())
                chunk_ids.append(doc_id)
                if end >= len(matches):
                    break
                start += step
    return chunks, chunk_ids


def _rrf_merge(dense_ranked: list, bm25_ranked: list, k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion over two ranked lists of chunk indices."""
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_ranked):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(bm25_ranked):
        scores[idx] = scores.get(idx, 0.0) + BM25_WEIGHT / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


class NLPManager:
    loaded = False

    def __init__(self):
        if not (_st_available and _faiss_available and _hf_available):
            raise RuntimeError(
                "sentence-transformers, faiss, and transformers are all required"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedder = SentenceTransformer(EMBED_MODEL, device=device)
        if hasattr(self.embedder, "max_seq_length"):
            self.embedder.max_seq_length = min(self.embedder.max_seq_length, 512)

        # Cross-encoder reranker — greatly improves top-3 precision for L3
        # cross-document and ambiguous questions.
        self.reranker = CrossEncoder(RERANKER_MODEL, device=device)

        # LLM: use 4-bit quantisation when bitsandbytes is available to halve
        # VRAM usage (~14 GB fp16 → ~4 GB int4) and speed up generation.
        self.device = device
        self.tokenizer = None
        self.llm = None
        self._llm_lock = threading.Lock()
        self.index = None
        self.bm25 = None
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []

    def _ensure_llm(self) -> None:
        """Loads the generator on first use so /health is available quickly."""
        if self.llm is not None:
            return

        with self._llm_lock:
            if self.llm is not None:
                return

            load_kwargs: dict = {"device_map": "auto"}
            if (
                USE_4BIT
                and _bnb_available
                and BitsAndBytesConfig is not None
                and self.device == "cuda"
            ):
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                load_kwargs["torch_dtype"] = (
                    torch.float16 if self.device == "cuda" else torch.float32
                )

            self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            self.llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL, **load_kwargs)
            self.llm.eval()

    def load_corpus(self, documents: list[dict[str, str]]) -> None:
        """Loads the corpus of documents for RAG QA."""
        texts = [d["document"] for d in documents]
        ids = [d["id"] for d in documents]

        chunks, chunk_ids = _chunk_documents(texts, ids)

        # --- Dense index (FAISS inner-product = cosine sim on normalised vecs) ---
        embeddings = self.embedder.encode(
            chunks,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

        # --- Sparse index (BM25) — improves recall for proper nouns / named entities ---
        if _bm25_available:
            tokenised = [_tokenize_for_sparse(c) for c in chunks]
            self.bm25 = BM25Okapi(tokenised)

        self.doc_ids = chunk_ids
        self.doc_texts = chunks
        self.loaded = True

    def qa(self, question: str) -> dict[str, list[str] | str]:
        """Performs RAG question answering.

        Returns:
            A dict with "documents" (list of up to 3 doc IDs) and "answer".
            L4 (no relevant docs): both empty.
            L5 (false premise): documents populated, answer empty.
        """
        # --- Step 1: Hybrid retrieval (dense + BM25) ---
        q_emb = self.embedder.encode(
            [_format_embed_query(question)],
            normalize_embeddings=True,
        )
        scores, indices = self.index.search(q_emb.astype(np.float32), TOP_K)
        dense_ranked = [int(i) for i in indices[0]]

        if self.bm25 is not None:
            bm25_scores = self.bm25.get_scores(_tokenize_for_sparse(question))
            bm25_ranked = list(np.argsort(bm25_scores)[::-1][:TOP_K])
            combined = _rrf_merge(dense_ranked, bm25_ranked)
        else:
            combined = dense_ranked
        combined = combined[:RERANK_TOP_K]

        # L4 guard: if the top dense score is below threshold, declare unanswerable
        max_dense_score = float(scores[0][0])
        if max_dense_score < UNANSWERABLE_THRESHOLD:
            return {"documents": [], "answer": ""}

        # --- Step 2: Rerank with cross-encoder ---
        candidate_texts = [self.doc_texts[i] for i in combined]
        pairs = [(question, t) for t in candidate_texts]
        rerank_scores = self.reranker.predict(pairs)
        reranked_order = np.argsort(rerank_scores)[::-1]

        # Deduplicate to top-3 unique doc IDs after reranking
        seen: set[str] = set()
        top3_ids: list[str] = []
        top3_texts: list[str] = []
        for pos in reranked_order:
            idx = combined[pos]
            doc_id = self.doc_ids[idx]
            if doc_id not in seen:
                seen.add(doc_id)
                top3_ids.append(doc_id)
                top3_texts.append(self.doc_texts[idx])
            if len(top3_ids) == 3:
                break

        # --- Step 3: Generate answer ---
        context = "\n\n".join(
            f"[{doc_id}]: {text}" for doc_id, text in zip(top3_ids, top3_texts)
        )
        answer = self._generate_answer(question, context)

        # L5: false premise detected by LLM — return doc IDs for partial credit
        if "UNANSWERABLE" in answer.upper():
            return {"documents": top3_ids, "answer": ""}

        return {"documents": top3_ids, "answer": answer}

    def _generate_answer(self, question: str, context: str) -> str:
        self._ensure_llm()
        user_content = (
            f"Documents:\n{context}\n\nQuestion: {question}\n"
            "Answer (or UNANSWERABLE):"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)
        with torch.no_grad():
            output = self.llm.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
