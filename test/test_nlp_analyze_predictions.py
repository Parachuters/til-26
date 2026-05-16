import importlib.util
from pathlib import Path


def _load_analyzer_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "nlp"
        / "analyze_predictions.py"
    )
    spec = importlib.util.spec_from_file_location("nlp_analyzer_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_classify_l4_requires_empty_documents_and_empty_answer():
    analyzer = _load_analyzer_module()
    gt = {"source_docs": [], "answer": None, "question": "What is outside Haven?"}

    correct = analyzer.classify_prediction(gt, {"documents": [], "answer": ""})
    wrong = analyzer.classify_prediction(gt, {"documents": ["DOC-1"], "answer": ""})

    assert correct["bucket"] == "l4_correct"
    assert correct["score_floor"] == 1.0
    assert wrong["bucket"] == "l4_false_positive"
    assert wrong["score_floor"] == 0.0


def test_classify_l5_requires_doc_overlap_and_empty_answer():
    analyzer = _load_analyzer_module()
    gt = {"source_docs": ["DOC-7"], "answer": "", "question": "False premise?"}

    correct = analyzer.classify_prediction(gt, {"documents": ["DOC-7"], "answer": ""})
    non_empty = analyzer.classify_prediction(
        gt, {"documents": ["DOC-7"], "answer": "A hallucinated answer."}
    )
    missed_doc = analyzer.classify_prediction(gt, {"documents": ["DOC-8"], "answer": ""})

    assert correct["bucket"] == "l5_correct"
    assert correct["score_floor"] == 1.0
    assert non_empty["bucket"] == "l5_non_empty_answer"
    assert non_empty["score_floor"] == 0.4
    assert missed_doc["bucket"] == "l5_doc_miss"
    assert missed_doc["score_floor"] == 0.0


def test_classify_answerable_splits_retrieval_miss_from_answer_check_needed():
    analyzer = _load_analyzer_module()
    gt = {"source_docs": ["DOC-2"], "answer": "Mara Voss", "question": "Who led it?"}

    hit = analyzer.classify_prediction(gt, {"documents": ["DOC-2"], "answer": "Mara"})
    miss = analyzer.classify_prediction(gt, {"documents": ["DOC-3"], "answer": "Mara"})

    assert hit["bucket"] == "answerable_retrieval_hit"
    assert hit["score_floor"] == 0.4
    assert miss["bucket"] == "answerable_doc_miss"
    assert miss["score_floor"] == 0.0
