import importlib.util
import numpy as np
import sys
import types
from pathlib import Path


def _load_nlp_manager_module():
    class _DummySentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyCrossEncoder:
        def __init__(self, *args, **kwargs):
            pass

    heavy_stubs = {
        "faiss": types.SimpleNamespace(),
        "rank_bm25": types.SimpleNamespace(BM25Okapi=object),
        "sentence_transformers": types.SimpleNamespace(
            SentenceTransformer=_DummySentenceTransformer,
            CrossEncoder=_DummyCrossEncoder,
        ),
        "transformers": types.SimpleNamespace(
            AutoModelForCausalLM=object,
            AutoTokenizer=object,
            BitsAndBytesConfig=object,
        ),
        "torch": types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            float16=object(),
            float32=object(),
            no_grad=lambda: None,
        ),
        "bitsandbytes": types.SimpleNamespace(),
    }
    previous = {name: sys.modules.get(name) for name in heavy_stubs}
    sys.modules.update(heavy_stubs)
    module_path = Path(__file__).resolve().parents[1] / "nlp" / "src" / "nlp_manager.py"
    spec = importlib.util.spec_from_file_location("nlp_manager_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def test_sparse_tokenizer_normalizes_punctuation_and_case():
    manager = _load_nlp_manager_module()

    tokens = manager._tokenize_for_sparse("Dr. Nyx-7's Haven: Sector_12 floods!")

    assert tokens == ["dr", "nyx", "7", "s", "haven", "sector_12", "floods"]


def test_chunk_documents_uses_word_windows_with_overlap_and_preserves_doc_ids():
    manager = _load_nlp_manager_module()
    text = "alpha beta gamma delta epsilon zeta eta theta iota"

    chunks, chunk_ids = manager._chunk_documents(
        [text],
        ["DOC-1"],
        max_words=4,
        overlap_words=1,
    )

    assert chunk_ids == ["DOC-1", "DOC-1", "DOC-1"]
    assert chunks == [
        "alpha beta gamma delta",
        "delta epsilon zeta eta",
        "eta theta iota",
    ]


def test_embed_query_text_adds_bge_instruction_for_plain_question():
    manager = _load_nlp_manager_module()

    query = manager._format_embed_query("Who governs Haven?")

    assert query.startswith("Represent this sentence for searching relevant passages:")
    assert query.endswith("Who governs Haven?")


def test_parse_verifier_json_accepts_valid_json_inside_model_output():
    manager = _load_nlp_manager_module()

    parsed = manager._parse_verifier_json(
        '```json\n{"status":"answerable",'
        '"evidence_quote":"Haven renews permits yearly.",'
        '"reason":"The quote answers the renewal cadence."}\n```'
    )

    assert parsed == {
        "status": "answerable",
        "evidence_quote": "Haven renews permits yearly.",
        "reason": "The quote answers the renewal cadence.",
    }


def test_parse_verifier_json_rejects_invalid_status_and_missing_answerable_quote():
    manager = _load_nlp_manager_module()

    assert manager._parse_verifier_json(
        '{"status":"maybe","evidence_quote":"Some quote.","reason":"Bad status."}'
    )["status"] == "insufficient_info"
    assert manager._parse_verifier_json(
        '{"status":"answerable","evidence_quote":"","reason":"No quote."}'
    )["status"] == "insufficient_info"
    assert manager._parse_verifier_json("not json")["status"] == "insufficient_info"


def test_clean_answer_removes_echoed_question_and_answer_prefix():
    manager = _load_nlp_manager_module()

    answer = manager._clean_answer(
        "At what occasion was Wampa Robotics destroyed in 2070?",
        "At what occasion was Wampa Robotics destroyed in 2070? Answer: In a ceremony.",
    )

    assert answer == "In a ceremony."


def test_clean_answer_only_removes_formatting_not_semantic_content():
    manager = _load_nlp_manager_module()

    answer = manager._clean_answer(
        "Where is the alternate permit stated?",
        "Answer: The document does not mention an alternate permit. [DOC-7]",
    )

    assert answer == "The document does not mention an alternate permit."


def test_select_evidence_snippet_prefers_sentences_matching_question_terms():
    manager = _load_nlp_manager_module()
    text = (
        "The casino opened under a provisional licence. "
        "Caulfield's Casino is registered under Class III permit SH-EV-00714. "
        "A later food inspection found no violations."
    )

    snippet = manager._select_evidence_snippet(
        "Under what license class and permit number is Caulfield's Casino registered?",
        text,
    )

    assert snippet == "Caulfield's Casino is registered under Class III permit SH-EV-00714."


class _FakeEmbedder:
    def __init__(self, vector):
        self.vector = vector

    def encode(self, texts, normalize_embeddings=True):
        return np.array([self.vector], dtype=np.float32)


class _FakeIndex:
    def __init__(self, score):
        self.score = score

    def search(self, q_emb, top_k):
        return (
            np.array(
                [[self.score, self.score - 0.05, self.score - 0.1]],
                dtype=np.float32,
            ),
            np.array([[0, 1, 2]], dtype=np.int64),
        )


class _FakeReranker:
    def predict(self, pairs):
        return np.array([0.9, 0.8, 0.7], dtype=np.float32)


def _fake_loaded_manager(module, dense_score=0.8, verifier_status="insufficient_info"):
    manager = module.NLPManager.__new__(module.NLPManager)
    manager.embedder = _FakeEmbedder([1.0, 0.0])
    manager.index = _FakeIndex(dense_score)
    manager.bm25 = None
    manager.reranker = _FakeReranker()
    manager.doc_ids = ["DOC-1", "DOC-2", "DOC-3"]
    manager.doc_texts = [
        "The launch permit was renewed annually by the CGC.",
        "A related operating memo described harbour inspections.",
        "The public archive lists no alternate permit process.",
    ]
    manager._verify_evidence = types.MethodType(
        lambda self, question, context, retrieval_signals: {
            "status": verifier_status,
            "evidence_quote": (
                "The launch permit was renewed annually by the CGC."
                if verifier_status == "answerable"
                else ""
            ),
            "reason": "test verifier",
        },
        manager,
    )
    manager._generate_answer = types.MethodType(
        lambda self, question, context, evidence_quote="": "annually",
        manager,
    )
    return manager


def test_qa_false_premise_returns_top_docs_with_empty_answer():
    module = _load_nlp_manager_module()
    manager = _fake_loaded_manager(module, verifier_status="false_premise")

    result = manager.qa("Which non-existent alternate permit did the CGC cite?")

    assert result == {"documents": ["DOC-1", "DOC-2", "DOC-3"], "answer": ""}


def test_qa_weak_retrieval_returns_empty_docs_without_verifier_call():
    module = _load_nlp_manager_module()
    manager = _fake_loaded_manager(module, dense_score=0.1)
    manager._verify_evidence = types.MethodType(
        lambda self, question, context, retrieval_signals: (_ for _ in ()).throw(
            AssertionError("verifier should not run for weak retrieval")
        ),
        manager,
    )

    result = manager.qa("What does an unrelated external standard require?")

    assert result == {"documents": [], "answer": ""}


def test_qa_strong_insufficient_info_returns_top_docs_with_empty_answer():
    module = _load_nlp_manager_module()
    manager = _fake_loaded_manager(
        module,
        dense_score=0.8,
        verifier_status="insufficient_info",
    )

    result = manager.qa("Which alternate permit was cited?")

    assert result == {"documents": ["DOC-1", "DOC-2", "DOC-3"], "answer": ""}
