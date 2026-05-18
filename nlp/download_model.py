"""Pre-downloads NLP models into the Docker image cache."""

import os

embed_model = os.getenv("NLP_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
reranker_model = os.getenv("NLP_RERANKER_MODEL", "BAAI/bge-reranker-base")
llm_model = os.getenv("NLP_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
use_reranker = os.getenv("NLP_USE_RERANKER", "0").lower() in {"1", "true", "yes", "on"}
use_llm = os.getenv("NLP_USE_LLM", "0").lower() in {"1", "true", "yes", "on"}

print(f"Downloading embedding model: {embed_model}")
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer(embed_model)

if use_reranker:
    print(f"Downloading reranker: {reranker_model}")
    CrossEncoder(reranker_model)

if use_llm:
    print(f"Downloading LLM: {llm_model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    AutoTokenizer.from_pretrained(llm_model)
    AutoModelForCausalLM.from_pretrained(llm_model, torch_dtype="auto")

print("All models downloaded.")
