import pandas as pd

from app.nlp.evaluate_model import elements_from_cell, substring_partial_overlap, evaluate_ner_model_partial_overlap, evaluate_summary_rouge_score
from app.nlp.utils import (
    compose_trial_text, chunk_text_by_chars, run_ner_on_long_text, clean_population_entities,
    merge_entities, extract_pico_from_merged_entities, normalize_intervention, is_substring_duplicate,
    deduplicate_intervention_entities, summarize_textRank, extract_comparator, remove_comparator_terms,
    clean_outcomes, process_trials_for_PICO
)

def test_elements_from_cell_with_string():
    assert elements_from_cell("DrugA; Drug-B , placebo") == {"druga", "drug-b", "placebo"}

def test_elements_from_cell_with_list():
    assert elements_from_cell([" Lanabecestat", "DrugB"]) == {"lanabecestat", "drugb"}

def test_elements_from_cell_with_nan():
    assert elements_from_cell(float('nan')) == set()
    assert elements_from_cell(pd.NA) == set()

def test_elements_from_cell_with_special_chars():
    # Remove punctuation, keep hyphens within words
    assert elements_from_cell(";Drug-X (oral)") == {"drug-x", "oral"}
    assert elements_from_cell("; DrugX*") == {"drugx"}

def test_substring_partial_overlap_simple():
    # Exact match should give perfect result
    gold, pred = {"lanabecestat"}, {"lanabecestat"}
    p, r, f1 = substring_partial_overlap(gold, pred)
    assert (p, r, f1) == (1, 1, 1)

def test_substring_partial_overlap_substrings():
    gold, pred = {"lanabecestat"}, {"as lanabecestat"}
    p, r, f1 = substring_partial_overlap(gold, pred)
    assert p == 1
    assert r == 1
    assert f1 == 1

    # Partial match: pred="drug", gold="drug treatment"
    gold, pred = {"drug treatment"}, {"drug"}
    p, r, f1 = substring_partial_overlap(gold, pred)
    assert p == 1    # 'drug' matches gold
    assert r == 1    # 'drug treatment' contains 'drug'
    assert f1 == 1

def test_substring_partial_overlap_no_match():
    gold, pred = {"foo"}, {"bar"}
    p, r, f1 = substring_partial_overlap(gold, pred)
    assert p == 0
    assert r == 0
    assert f1 == 0

def test_substring_partial_overlap_empty():
    assert substring_partial_overlap(set(), set()) == (1, 1, 1)
    assert substring_partial_overlap({"a"}, set()) == (0, 0, 0)
    assert substring_partial_overlap(set(), {"a"}) == (0, 0, 0)

def test_evaluate_ner_model_partial_overlap_smoke():
    df_gold = pd.DataFrame({
        'intervention_extracted': ["Lanabecestat; placebo", "Drug B"],
        'population_extracted': ["men; adults", "patients"]
    })
    df_model = pd.DataFrame({
        'intervention_extracted': ["as lanabecestat", "DrugB"],
        'population_extracted': ["adult males", "patients"]
    })
    pico = ["intervention_extracted", "population_extracted"]
    df_eval, summary = evaluate_ner_model_partial_overlap(df_gold.copy(), df_model, pico, add_rouge=False)
    # Only PICO partial columns
    for col in pico:
        for metric in ["partial_precision", "partial_recall", "partial_f1"]:
            assert f"{col}_{metric}" in df_eval.columns
    # Do NOT check for ROUGE columns here, since no summary columns present!

def test_evaluate_ner_model_partial_overlap_with_summary():
    # Example reference and model summaries
    df_gold = pd.DataFrame({
        'short_description': [
            "Lanabecestat reduces amyloid-beta in adults with Alzheimer's.",
            "Drug B trial in patients."
        ],
        'intervention_extracted': ["Lanabecestat", "Drug B"],
        'population_extracted': ["adults", "patients"]
    })
    df_model = pd.DataFrame({
        'summary': [
            "Lanabecestat helps adults with Alzheimer's.",
            "Drug B used in trial patients."
        ],
        'intervention_extracted': ["Lanabecestat", "Drug B"],
        'population_extracted': ["adults", "patients"]
    })
    pico = ["intervention_extracted", "population_extracted"]
    # Run with summary columns specified and add_rouge=True
    df_eval, summary = evaluate_ner_model_partial_overlap(
        df_gold.copy(), df_model, pico,
        summary_gold_col='short_description',
        summary_pred_col='summary',
        add_rouge=True
    )
    # Check that ROUGE rows are present and within [0,1]
    for metric in ["SUMMARY_ROUGE-1", "SUMMARY_ROUGE-2", "SUMMARY_ROUGE-L"]:
        assert metric in summary.index
        for k in ["precision", "recall", "f1"]:
            v = summary.loc[metric, k]
            assert 0 <= v <= 1, f"{metric} {k} out of range: {v}"

def test_evaluate_ner_model_partial_overlap_with_rouge():
    df_gold = pd.DataFrame({
        'summary_reference': [
            "DrugA reduces symptoms in adults.",
            "DrugB trial in male patients."
        ],
        'intervention_extracted': ["DrugA", "DrugB"],
        'population_extracted': ["adults", "male patients"]
    })
    df_model = pd.DataFrame({
        'summary': [
            "DrugA helps adults with symptoms.",
            "DrugB used in male patients."
        ],
        'intervention_extracted': ["DrugA", "DrugB"],
        'population_extracted': ["adults", "male patients"]
    })
    pico = ["intervention_extracted", "population_extracted"]
    df_eval, summary = evaluate_ner_model_partial_overlap(
        df_gold.copy(), df_model, pico,
        summary_gold_col='summary_reference', summary_pred_col='summary'
    )
    # Now check for ROUGE columns as well!
    for rn in ['rouge1', 'rouge2', 'rougeL']:
        for metric in ["precision", "recall", "f1"]:
            colname = f"summary_{rn}_{metric}"
            assert colname in df_eval.columns
            assert df_eval[colname].between(0, 1).all()
    # Check metrics table includes ROUGE rows
    for rn_name in ["SUMMARY_ROUGE-1", "SUMMARY_ROUGE-2", "SUMMARY_ROUGE-L"]:
        assert rn_name in summary.index
        for metric in ["precision", "recall", "f1"]:
            val = summary.loc[rn_name, metric]
            assert 0 <= val <= 1