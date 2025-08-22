import os
import pandas as pd
import re
import sys
# Insert the project root (biomed_extractor) into sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# load functions for import of clinicaltrials.gov data written previously
from app.data.loader import load_trials_json, extract_from_clinicaltrials
#from utils import * # custom functions required for NER and summarization
from app.nlp.pipelines import load_ner_pipeline
from app.nlp.utils import (
    compose_trial_text, chunk_text_by_chars, run_ner_on_long_text, clean_population_entities,
    merge_entities, extract_pico_from_merged_entities, normalize_intervention, is_substring_duplicate,
    deduplicate_intervention_entities, summarize_textRank, extract_comparator, remove_comparator_terms,
    clean_outcomes, process_trials_for_PICO
)


def elements_from_cell(cell):
    """
    Convert a DataFrame cell to a set of clean, lowercased elements.

    Cells are split by semicolon and comma (if string), or iterated if they are lists.
    Each split part is further split by whitespace after cleaning punctuation (except hyphens).
    Leading/trailing whitespace is trimmed, and empty elements are removed.

    Args:
        cell (Any): A DataFrame cell, expected to be either a list, a semicolon/comma-separated string, or NaN.

    Returns:
        set of str: Set of cleaned, lowercased, deduplicated textual elements.
    """
    import numpy as np
    # Check arraylike/Series and convert to list first
    if isinstance(cell, (pd.Series, np.ndarray)):
        cell = list(cell)

    # Now check for NA only if cell is not a list
    if not isinstance(cell, list) and pd.isna(cell):
        return set()

    if isinstance(cell, list):
        raw_parts = []
        for x in cell:
            raw_parts.extend(re.split(r'[;,]', str(x)))
    else:
        raw_parts = re.split(r'[;,]', str(cell))
    out = set()
    for part in raw_parts:
        cleaned = re.sub(r"[^\w\s-]", " ", part.lower())
        for token in cleaned.split():
            if token:
                out.add(token)
    return set(map(str, out))


def substring_partial_overlap(set_gold, set_pred):
    """
    Compute partial overlap metrics between two sets of strings,
    counting as a match if one is a substring of the other (in either direction).

    Args:
        set_gold (set of str): The set of gold-standard entities for a trial/field.
        set_pred (set of str): The set of predicted entities for the same trial/field.

    Returns:
        tuple: (precision, recall, f1), where:
            - precision = fraction of predicted items matching any gold item by substring
            - recall = fraction of gold items matched by any prediction by substring
            - f1 = harmonic mean of precision and recall (0 if both are 0)
    """
    if not set_gold and not set_pred:
        return 1.0, 1.0, 1.0
    match_gold = sum(
        any((g in p or p in g) and g and p for p in set_pred) 
        for g in set_gold
    )
    match_pred = sum(
        any((p in g or g in p) and g and p for g in set_gold) 
        for p in set_pred
    )
    precision = match_pred / len(set_pred) if set_pred else 0.0
    recall = match_gold / len(set_gold) if set_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def add_partial_overlap_cols(df_gold, ner_res_model, pico_cols):
    """
    For each PICO column, compares gold and predicted values row-by-row
    using substring-overlap matching, and adds per-row precision, recall, and F1 columns
    to df_gold for detailed error analysis. Also returns a summary DataFrame of averages.

    Args:
        df_gold (pd.DataFrame): The DataFrame with gold-standard PICO values.
        ner_res_model (pd.DataFrame): The DataFrame with predicted PICO values (must have same columns, matching order).
        pico_cols (list of str): List of column names (e.g. ["intervention_extracted", ...])
    
    Returns:
        tuple:
            - pd.DataFrame: df_gold with new columns for each evaluated PICO field:
                "{col}_partial_precision", "{col}_partial_recall", "{col}_partial_f1"
            - pd.DataFrame: summary table with mean precision, recall, and F1 for each field.
    """
    for col in pico_cols:
        partial_precisions, partial_recalls, partial_f1s = [], [], []
        if col not in ner_res_model.columns:
            df_gold[f"{col}_partial_precision"] = None
            df_gold[f"{col}_partial_recall"] = None
            df_gold[f"{col}_partial_f1"] = None
            continue
        for gold_cell, pred_cell in zip(df_gold[col], ner_res_model[col]):
            gold_elems = elements_from_cell(gold_cell)
            pred_elems = elements_from_cell(pred_cell)
            p, r, f1 = substring_partial_overlap(gold_elems, pred_elems)
            partial_precisions.append(p)
            partial_recalls.append(r)
            partial_f1s.append(f1)
        df_gold[f"{col}_partial_precision"] = partial_precisions
        df_gold[f"{col}_partial_recall"] = partial_recalls
        df_gold[f"{col}_partial_f1"] = partial_f1s

    evaluation_table = pd.DataFrame(
        {
            "precision":   [df_gold[f"{col}_partial_precision"].mean() for col in pico_cols],
            "recall":      [df_gold[f"{col}_partial_recall"].mean()    for col in pico_cols],
            "f1":          [df_gold[f"{col}_partial_f1"].mean()        for col in pico_cols]
        },
        index=[col.replace("_extracted", "") for col in pico_cols]
    )
    
    return df_gold, evaluation_table

if __name__ == "__main__":
    # Get the PROJECT ROOT (biomed-extractor/)
    PROJECT_ROOT = 'c:\\Users\\elena.jolkver\\Documents\\github\\biomed_extractor'
    # Data directory at top level
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data\\annotated')
    # load data to extract info from
    df_json = load_trials_json(filepath = DATA_DIR, filename ='ctg-studies_for_gold.json')
    mydf_manual_annotation = extract_from_clinicaltrials(df_json)

    # load gold-standard results to evaluate results on
    df_gold = load_trials_json(filepath = DATA_DIR, filename ='gold_standard.json')
    for col in ["population", "intervention","comparator", "outcome"]:
        df_gold[col] = df_gold[col].apply(
            lambda x: '; '.join(str(e) for e in x) if isinstance(x, list) else x
        )
        # set to lowercase (to align with automatic extraction)
        df_gold[col] = df_gold[col].str.lower()
    df_gold.sort_values(by=['doc_id'], inplace=True)
    df_gold["intervention"] = df_gold["intervention"].apply(normalize_intervention)

    # process test file for PICO elements
    ner_pipeline = load_ner_pipeline()
    ner_res_model = process_trials_for_PICO(mydf_manual_annotation, ner_pipeline)
    ner_res_model.sort_values(by=['nctId'], inplace=True)
    # rename columns to match gold standard
    ner_res_model.rename(columns={'population_extracted': 'population',
                                'intervention_extracted': 'intervention',
                                'comparator_extracted': 'comparator',
                                'outcome_extracted': 'outcome',
                                'summary_extracted': 'summary'}, inplace=True)
    
    # evaluate results
    pico_cols = ["population", "intervention", "comparator", "outcome"]
    df_gold_with_partial, evaluation_table = add_partial_overlap_cols(df_gold.copy(), ner_res_model, pico_cols)
    print(evaluation_table)
    
