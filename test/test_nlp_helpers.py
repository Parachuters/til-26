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

    assert tokens == [
        "dr",
        "nyx",
        "7",
        "s",
        "haven",
        "sector_12",
        "sector",
        "12",
        "floods",
    ]


def test_sparse_tokenizer_preserves_lore_compounds_codes_and_parts():
    manager = _load_nlp_manager_module()

    tokens = manager._tokenize_for_sparse(
        "Nyx-7 cited SH-EV-00714 for CGC Sector_12 operations."
    )

    assert tokens == [
        "nyx-7",
        "nyx",
        "7",
        "cited",
        "sh-ev-00714",
        "sh",
        "ev",
        "00714",
        "for",
        "cgc",
        "sector_12",
        "sector",
        "12",
        "operations",
    ]


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


def test_time_first_defaults_use_extractive_pipeline_without_heavy_qa_models():
    manager = _load_nlp_manager_module()

    assert manager.PIPELINE_MODE == "extractive"
    assert manager.USE_RERANKER is False
    assert manager.USE_LLM is False


def test_build_corpus_lexicon_extracts_lore_entities_codes_and_context():
    manager = _load_nlp_manager_module()

    lexicon = manager._build_corpus_lexicon(
        [
            (
                "The Cascade flooded old governments. "
                "CGC assigned SH-EV-00714 to Nyx-7 under the Blackshore Accords."
            )
        ],
        ["DOC-1"],
    )

    assert "the cascade" in lexicon
    assert "blackshore accords" in lexicon
    assert "cgc" in lexicon
    assert "sh-ev-00714" in lexicon
    assert "nyx-7" in lexicon
    assert lexicon["the cascade"]["surface"] == "The Cascade"
    assert lexicon["the cascade"]["doc_ids"] == {"DOC-1"}
    assert "flooded old governments" in lexicon["the cascade"]["contexts"][0]


def test_expand_query_for_retrieval_appends_matching_entities_and_context_terms():
    manager = _load_nlp_manager_module()
    lexicon = manager._build_corpus_lexicon(
        [
            (
                "The Cascade flooded old governments. "
                "Haven survived as the last great city."
            ),
            "Blackshore Accords govern maritime approaches to Haven.",
        ],
        ["DOC-1", "DOC-2"],
    )

    expanded = manager._expand_query_for_retrieval(
        "Which flood left the last great city under Haven control?",
        lexicon,
        limit=4,
    )

    assert expanded.startswith(
        "Which flood left the last great city under Haven control?"
    )
    assert "The Cascade" in expanded
    assert "Haven" in expanded


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


def test_parse_qa_json_accepts_answerable_payload_inside_model_output():
    manager = _load_nlp_manager_module()

    parsed = manager._parse_qa_json(
        '```json\n{"status":"answerable",'
        '"evidence_quote":"The launch permit was renewed annually by the CGC.",'
        '"answer":"annually","reason":"The quote gives the cadence."}\n```'
    )

    assert parsed == {
        "status": "answerable",
        "evidence_quote": "The launch permit was renewed annually by the CGC.",
        "answer": "annually",
        "reason": "The quote gives the cadence.",
    }


def test_parse_qa_json_rejects_invalid_payloads_to_empty_insufficient_info():
    manager = _load_nlp_manager_module()

    assert manager._parse_qa_json("not json") == {
        "status": "insufficient_info",
        "evidence_quote": "",
        "answer": "",
        "reason": "No JSON object found.",
    }
    assert manager._parse_qa_json(
        '{"status":"answerable","evidence_quote":"","answer":"annually"}'
    )["status"] == "insufficient_info"
    assert manager._parse_qa_json(
        '{"status":"insufficient_info","evidence_quote":"","answer":"annually"}'
    )["answer"] == ""


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


def test_extract_salient_claims_finds_entities_numbers_quotes_and_markers():
    manager = _load_nlp_manager_module()

    claims = manager._extract_salient_claims(
        "Other than TEC, which CGC report quoted 'red harbour permit' at 17.5% in Q4 2076?"
    )

    assert {"tec", "cgc"}.issubset(claims["entities"])
    assert "red harbour permit" in claims["quoted"]
    assert {"17.5%", "q4", "2076"}.issubset(claims["numbers"])
    assert "other than" in claims["markers"]


def test_extract_salient_claims_drops_question_prefix_from_entity():
    manager = _load_nlp_manager_module()

    claims = manager._extract_salient_claims(
        "What motivated Phyrexis Group to publicly claim credit?"
    )

    assert "phyrexis group" in claims["entities"]
    assert "what motivated phyrexis group" not in claims["entities"]

    yes_no_claims = manager._extract_salient_claims(
        "Did Phyrexis Group publicly claim credit?"
    )

    assert "phyrexis group" in yes_no_claims["entities"]
    assert "did phyrexis group" not in yes_no_claims["entities"]
    assert "phyrexis" not in yes_no_claims["entities"]


def test_verify_question_support_false_premise_always_vetoes_answer():
    manager = _load_nlp_manager_module()

    decision = manager._verify_question_support(
        "Which non-existent alternate permit did the CGC cite?",
        "[DOC-1]: The CGC cited SH-EV-00714.",
        {"status": "false_premise", "evidence_quote": "", "reason": "test"},
    )

    assert decision == "contradicted_or_unsupported"


def test_verify_question_support_vetoes_unsupported_required_entity():
    manager = _load_nlp_manager_module()

    decision = manager._verify_question_support(
        "Which regulatory body other than TEC classified the breach?",
        "[DOC-1]: TEC classified the breach after the audit.",
        {"status": "insufficient_info", "evidence_quote": "", "reason": "test"},
    )

    assert decision == "contradicted_or_unsupported"


def test_verify_question_support_vetoes_unsupported_required_number():
    manager = _load_nlp_manager_module()

    decision = manager._verify_question_support(
        "Which permit was renewed in 2077?",
        "[DOC-1]: SH-EV-00714 was renewed in 2076 by the CGC.",
        {"status": "insufficient_info", "evidence_quote": "", "reason": "test"},
    )

    assert decision == "contradicted_or_unsupported"


def test_verify_question_support_vetoes_negated_evidence_for_positive_premise():
    manager = _load_nlp_manager_module()

    decision = manager._verify_question_support(
        "Did Phyrexis Group publicly claim credit for funding the event?",
        "[DOC-1]: Phyrexis Group made no public claim of credit for the event.",
        {"status": "insufficient_info", "evidence_quote": "", "reason": "test"},
    )

    assert decision == "contradicted_or_unsupported"


def test_verify_question_support_keeps_answerable_overlap_unknown():
    manager = _load_nlp_manager_module()

    decision = manager._verify_question_support(
        "Which permit did the CGC renew?",
        "[DOC-1]: The CGC renewed SH-EV-00714 annually.",
        {"status": "insufficient_info", "evidence_quote": "", "reason": "test"},
    )

    assert decision == "unknown"


def test_verify_question_support_marks_answerable_as_supported():
    manager = _load_nlp_manager_module()

    decision = manager._verify_question_support(
        "Which permit did the CGC renew?",
        "[DOC-1]: The CGC renewed SH-EV-00714 annually.",
        {
            "status": "answerable",
            "evidence_quote": "The CGC renewed SH-EV-00714 annually.",
            "reason": "test",
        },
    )

    assert decision == "supported"


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


class _FakeBM25:
    def __init__(self, scores):
        self.scores = np.array(scores, dtype=np.float32)
        self.queries = []

    def get_scores(self, tokens):
        self.queries.append(tokens)
        return self.scores


class _FakeReranker:
    def predict(self, pairs):
        return np.array([0.9, 0.8, 0.7], dtype=np.float32)


def _fake_loaded_manager(
    module,
    dense_score=0.8,
    qa_status="insufficient_info",
    bm25_scores=None,
    lexicon=None,
):
    manager = module.NLPManager.__new__(module.NLPManager)
    manager.embedder = _FakeEmbedder([1.0, 0.0])
    manager.index = _FakeIndex(dense_score)
    manager.bm25 = _FakeBM25(bm25_scores) if bm25_scores is not None else None
    manager.reranker = _FakeReranker()
    manager.pipeline_mode = "llm"
    manager.use_reranker = True
    manager.use_llm = True
    manager.lexicon = lexicon or {}
    manager.doc_ids = ["DOC-1", "DOC-2", "DOC-3"]
    manager.doc_texts = [
        "The launch permit was renewed annually by the CGC.",
        "A related operating memo described harbour inspections.",
        "The public archive lists no alternate permit process.",
    ]
    manager._answer_from_evidence = types.MethodType(
        lambda self, question, context, retrieval_signals: {
            "status": qa_status,
            "evidence_quote": (
                "The launch permit was renewed annually by the CGC."
                if qa_status == "answerable"
                else ""
            ),
            "answer": "annually" if qa_status == "answerable" else "",
            "reason": "test verifier",
        },
        manager,
    )
    return manager


def test_qa_false_premise_returns_top_docs_with_empty_answer():
    module = _load_nlp_manager_module()
    manager = _fake_loaded_manager(module, qa_status="false_premise")

    result = manager.qa("Which non-existent alternate permit did the CGC cite?")

    assert result == {"documents": ["DOC-1", "DOC-2", "DOC-3"], "answer": ""}


def test_qa_weak_retrieval_returns_empty_docs_without_verifier_call():
    module = _load_nlp_manager_module()
    manager = _fake_loaded_manager(module, dense_score=0.1)
    manager._answer_from_evidence = types.MethodType(
        lambda self, question, context, retrieval_signals: (_ for _ in ()).throw(
            AssertionError("LLM answer should not run for weak retrieval")
        ),
        manager,
    )

    result = manager.qa("What does an unrelated external standard require?")

    assert result == {"documents": [], "answer": ""}


def test_qa_external_l4_question_returns_empty_docs_without_verifier_call():
    module = _load_nlp_manager_module()
    manager = _fake_loaded_manager(module, dense_score=0.8)
    manager._answer_from_evidence = types.MethodType(
        lambda self, question, context, retrieval_signals: (_ for _ in ()).throw(
            AssertionError("LLM answer should not run for external L4 questions")
        ),
        manager,
    )

    result = manager.qa(
        "What is the current FDA approval status for human somatic gene therapy "
        "treatments as of 2024?"
    )

    assert result == {"documents": [], "answer": ""}


def test_qa_low_dense_score_continues_when_bm25_entity_evidence_is_strong():
    module = _load_nlp_manager_module()
    lexicon = module._build_corpus_lexicon(
        ["CGC renewed the launch permit annually under SH-EV-00714."],
        ["DOC-1"],
    )
    manager = _fake_loaded_manager(
        module,
        dense_score=0.1,
        qa_status="answerable",
        bm25_scores=[8.0, 0.0, 0.0],
        lexicon=lexicon,
    )

    result = manager.qa("How often did CGC renew SH-EV-00714?")

    assert result == {"documents": ["DOC-1", "DOC-2", "DOC-3"], "answer": "annually"}
    assert "sh-ev-00714" in manager.bm25.queries[0]


def test_qa_answerable_combined_result_returns_clean_answer():
    module = _load_nlp_manager_module()
    manager = _fake_loaded_manager(
        module,
        dense_score=0.8,
        qa_status="answerable",
    )

    result = manager.qa("Which launch permit did the CGC renew?")

    assert result == {"documents": ["DOC-1", "DOC-2", "DOC-3"], "answer": "annually"}


def test_qa_unsupported_insufficient_info_returns_empty_answer():
    module = _load_nlp_manager_module()
    manager = _fake_loaded_manager(
        module,
        dense_score=0.8,
        qa_status="insufficient_info",
    )

    result = manager.qa(
        "Which vessel type was built at the yard, given that one was a frigate?"
    )

    assert result == {"documents": ["DOC-1", "DOC-2", "DOC-3"], "answer": ""}


def test_qa_uses_one_combined_llm_answer_method():
    module = _load_nlp_manager_module()
    manager = _fake_loaded_manager(module, qa_status="answerable")
    calls = {"combined": 0}

    def answer_once(self, question, context, retrieval_signals):
        calls["combined"] += 1
        return {
            "status": "answerable",
            "evidence_quote": "The launch permit was renewed annually by the CGC.",
            "answer": "annually",
            "reason": "test verifier",
        }

    manager._answer_from_evidence = types.MethodType(answer_once, manager)
    manager._verify_evidence = types.MethodType(
        lambda self, question, context, retrieval_signals: (_ for _ in ()).throw(
            AssertionError("separate verifier should not be called")
        ),
        manager,
    )
    manager._generate_answer = types.MethodType(
        lambda self, question, context, evidence_quote="": (_ for _ in ()).throw(
            AssertionError("separate generator should not be called")
        ),
        manager,
    )

    result = manager.qa("Which launch permit did the CGC renew?")

    assert result == {"documents": ["DOC-1", "DOC-2", "DOC-3"], "answer": "annually"}
    assert calls == {"combined": 1}


def test_qa_extractive_mode_skips_reranker_and_llm():
    module = _load_nlp_manager_module()
    manager = _fake_loaded_manager(module, dense_score=0.8)
    manager.pipeline_mode = "extractive"
    manager.use_reranker = False
    manager.use_llm = False
    manager.reranker = types.SimpleNamespace(
        predict=lambda pairs: (_ for _ in ()).throw(
            AssertionError("reranker should not run in extractive mode")
        )
    )
    manager._answer_from_evidence = types.MethodType(
        lambda self, question, context, retrieval_signals: (_ for _ in ()).throw(
            AssertionError("LLM answer should not run in extractive mode")
        ),
        manager,
    )

    result = manager.qa("How often did the CGC renew the launch permit?")

    assert result == {
        "documents": ["DOC-1", "DOC-2", "DOC-3"],
        "answer": "The launch permit was renewed annually by the CGC.",
    }
