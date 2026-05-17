"""Manages the NLP model."""

import json
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
VERIFIER_MAX_NEW_TOKENS = int(os.getenv("NLP_VERIFIER_MAX_NEW_TOKENS", "128"))
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
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", flags=re.UNICODE)
_VALID_VERIFIER_STATUSES = {"answerable", "false_premise", "insufficient_info"}
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "does", "for",
    "from", "given", "had", "has", "have", "how", "if", "in", "is", "it",
    "of", "on", "or", "that", "the", "their", "this", "to", "under", "was",
    "were", "what", "when", "where", "which", "who", "why", "with",
}

# System prompt grounded in the Clairos fictional setting so the LLM does not
# confuse in-world terminology with real-world knowledge.
SYSTEM_PROMPT = (
    "You are a precise QA assistant for the world of Clairos — a cyberpunk "
    "setting after the Cascade flood where megacorporations rule Haven, the "
    "last great city. Answer questions using ONLY the provided document "
    "excerpts.\n"
    "Rules:\n"
    "1. Return only the final answer, not the question or explanation.\n"
    "2. Copy names, identifiers, percentages, dates, and numbers exactly as "
    "written in the documents.\n"
    "3. If the question has a false premise, contradicts the documents, or no "
    "answer exists in the "
    "documents, respond with exactly: UNANSWERABLE\n"
    "4. Do not use outside knowledge or hallucinate facts not in the context."
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


def _content_words(text: str) -> set[str]:
    return {
        token
        for token in _tokenize_for_sparse(text)
        if token not in _STOPWORDS and len(token) > 1
    }


def _select_evidence_snippet(
    question: str,
    text: str,
    max_chars: int = 900,
    max_sentences: int = 1,
) -> str:
    """Keep the most question-relevant sentences from a retrieved chunk."""
    sentences = [m.group(0).strip() for m in _SENTENCE_RE.finditer(text)]
    sentences = [s for s in sentences if s]
    if not sentences:
        return text[:max_chars].strip()

    query_terms = _content_words(question)
    scored = []
    for index, sentence in enumerate(sentences):
        terms = _content_words(sentence)
        score = len(query_terms.intersection(terms))
        scored.append((score, index, sentence))

    selected = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_sentences]
    selected = sorted(selected, key=lambda item: item[1])
    snippet = " ".join(sentence for _, _, sentence in selected).strip()
    return snippet[:max_chars].strip()


def _clean_answer(question: str, answer: str) -> str:
    """Remove answer formatting without making semantic unanswerable decisions."""
    cleaned = answer.strip().strip('"')

    escaped_question = re.escape(question.strip())
    cleaned = re.sub(
        rf"^{escaped_question}\s*(?:Answer\s*:)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^(?:Answer|Final answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^The answer is\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\[[^\]]+\]\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _default_verifier_result(
    reason: str = "Verifier output could not be trusted.",
) -> dict[str, str]:
    return {"status": "insufficient_info", "evidence_quote": "", "reason": reason}


def _parse_verifier_json(text: str) -> dict[str, str]:
    """Parse and validate the verifier's strict JSON response."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return _default_verifier_result("No JSON object found.")

    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return _default_verifier_result("Invalid JSON.")

    if not isinstance(parsed, dict):
        return _default_verifier_result("JSON output was not an object.")

    status = str(parsed.get("status", "")).strip().lower()
    evidence_quote = str(parsed.get("evidence_quote", "")).strip()
    reason = str(parsed.get("reason", "")).strip()
    if status not in _VALID_VERIFIER_STATUSES:
        return _default_verifier_result("Invalid verifier status.")
    if status == "answerable" and not evidence_quote:
        return _default_verifier_result("Answerable status lacked an evidence quote.")

    return {
        "status": status,
        "evidence_quote": evidence_quote,
        "reason": reason,
    }


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
        dense_ranked = [int(i) for i in indices[0] if int(i) >= 0]
        if not dense_ranked:
            return {"documents": [], "answer": ""}

        max_dense_score = float(scores[0][0])
        if max_dense_score < UNANSWERABLE_THRESHOLD:
            return {"documents": [], "answer": ""}

        if self.bm25 is not None:
            bm25_scores = self.bm25.get_scores(_tokenize_for_sparse(question))
            bm25_ranked = list(np.argsort(bm25_scores)[::-1][:TOP_K])
            combined = _rrf_merge(dense_ranked, bm25_ranked)
        else:
            combined = dense_ranked
        combined = combined[:RERANK_TOP_K]
        if not combined:
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

        if not top3_ids:
            return {"documents": [], "answer": ""}

        # --- Step 3: Verify evidence before generating an answer ---
        context = "\n\n".join(
            f"[{doc_id}]: {_select_evidence_snippet(question, text, max_sentences=3)}"
            for doc_id, text in zip(top3_ids, top3_texts)
        )
        verifier = self._verify_evidence(
            question,
            context,
            {
                "max_dense_score": max_dense_score,
                "top_rerank_score": (
                    float(np.max(rerank_scores)) if len(rerank_scores) else 0.0
                ),
                "candidate_count": len(combined),
            },
        )
        if verifier["status"] == "false_premise":
            return {"documents": top3_ids, "answer": ""}
        if verifier["status"] == "insufficient_info":
            return {"documents": top3_ids, "answer": ""}

        # --- Step 4: Generate answer from verified evidence ---
        answer = self._generate_answer(question, context, verifier["evidence_quote"])
        answer = _clean_answer(question, answer)

        if answer.upper() == "UNANSWERABLE":
            return {"documents": top3_ids, "answer": ""}

        return {"documents": top3_ids, "answer": answer}

    def _verify_evidence(
        self,
        question: str,
        context: str,
        retrieval_signals: dict[str, float | int],
    ) -> dict[str, str]:
        """Ask the LLM whether retrieved snippets answer the exact question."""
        self._ensure_llm()
        user_content = (
            "Decide whether the context answers the exact question. "
            "Use only the context, not outside knowledge.\n\n"
            "Return only JSON with this shape:\n"
            '{"status":"answerable|false_premise|insufficient_info",'
            '"evidence_quote":"Exact sentence from context or empty string",'
            '"reason":"One short sentence"}\n\n'
            "Status rules:\n"
            "- answerable: the context directly answers every part of the question. "
            "evidence_quote must be an exact quote from the context.\n"
            "- false_premise: the context is relevant and contradicts a premise in the question.\n"
            "- insufficient_info: context is unrelated or lacks the specific answer.\n\n"
            "Example 1\n"
            "Question: How often are Haven permits renewed?\n"
            "Context: [D1]: Haven permits are renewed annually by the CGC.\n"
            "JSON: {\"status\":\"answerable\","
            "\"evidence_quote\":\"Haven permits are renewed annually by the CGC.\","
            "\"reason\":\"The quote gives the renewal frequency.\"}\n\n"
            "Example 2\n"
            "Question: Which blue permit did the CGC cite for the harbour?\n"
            "Context: [D2]: The CGC cited a red harbour permit. No blue permit was issued.\n"
            "JSON: {\"status\":\"false_premise\","
            "\"evidence_quote\":\"No blue permit was issued.\","
            "\"reason\":\"The context contradicts the requested blue permit.\"}\n\n"
            "Example 3\n"
            "Question: Who signed the alternate harbour permit?\n"
            "Context: [D3]: The harbour memo described inspection routes.\n"
            "JSON: {\"status\":\"insufficient_info\",\"evidence_quote\":\"\",\"reason\":\"The signer is not stated.\"}\n\n"
            f"Retrieval signals: {json.dumps(retrieval_signals, sort_keys=True)}\n"
            f"Question: {question}\n"
            f"Context:\n{context}\n"
            "JSON:"
        )
        messages = [
            {"role": "system", "content": "You are a strict evidence verifier."},
            {"role": "user", "content": user_content},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)
        with torch.no_grad():
            output = self.llm.generate(
                **inputs,
                max_new_tokens=VERIFIER_MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        verifier_text = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()
        return _parse_verifier_json(verifier_text)

    def _generate_answer(
        self,
        question: str,
        context: str,
        evidence_quote: str = "",
    ) -> str:
        self._ensure_llm()
        user_content = (
            f"Verified evidence quote:\n{evidence_quote}\n\n"
            f"Supporting snippets:\n{context}\n\n"
            f"Question: {question}\n"
            "Return only the shortest correct answer phrase. "
            "Do not repeat the question and do not include citations.\nAnswer:"
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
