import importlib.util
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


def test_external_current_events_guard_flags_l4_real_world_questions():
    manager = _load_nlp_manager_module()

    assert manager._looks_like_external_l4(
        "What is the current FDA approval status for human somatic gene therapy "
        "treatments as of 2024?"
    )
    assert not manager._looks_like_external_l4(
        "How often must Havenite residency documentation be renewed under CGC policy?"
    )


def test_clean_answer_removes_echoed_question_and_answer_prefix():
    manager = _load_nlp_manager_module()

    answer = manager._clean_answer(
        "At what occasion was Wampa Robotics destroyed in 2070?",
        "At what occasion was Wampa Robotics destroyed in 2070? Answer: In a ceremony.",
    )

    assert answer == "In a ceremony."


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
