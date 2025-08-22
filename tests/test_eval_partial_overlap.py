import pandas as pd

from app.nlp.evaluate_model import elements_from_cell, substring_partial_overlap, add_partial_overlap_cols
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

def test_add_partial_overlap_cols_smoke():
    # Minimal two-dataframe test
    df_gold = pd.DataFrame({
        'intervention_extracted': ["Lanabecestat; placebo", "Drug B"],
        'population_extracted': ["men; adults", "patients"]
    })
    df_model = pd.DataFrame({
        'intervention_extracted': ["as lanabecestat", "DrugB"],
        'population_extracted': ["adult males", "patients"]
    })
    pico = ["intervention_extracted", "population_extracted"]
    df_eval, summary = add_partial_overlap_cols(df_gold.copy(), df_model, pico)
    # Columns created
    for col in pico:
        assert f"{col}_partial_precision" in df_eval.columns
        assert f"{col}_partial_recall" in df_eval.columns
        assert f"{col}_partial_f1" in df_eval.columns
    # Summary is dataframe with correct index
    assert set(summary.index) == {"intervention", "population"}
    assert all(0.0 <= val <= 1.0 for val in summary["precision"])