"""Pre-downloads NLP models into the Docker image cache."""

import os

embed_model = os.getenv("NLP_EMBED_MODEL", "BAAI/bge-m3")
reranker_model = os.getenv("NLP_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
llm_model = os.getenv("NLP_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

print(f"Downloading embedding model: {embed_model}")
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer(embed_model)

print(f"Downloading reranker: {reranker_model}")
CrossEncoder(reranker_model)

print(f"Downloading LLM: {llm_model}")
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained(llm_model)
AutoModelForCausalLM.from_pretrained(llm_model, torch_dtype="auto")

print("All models downloaded.")
