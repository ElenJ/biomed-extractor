import os
import pandas as pd
import re
import sys
from rouge_score import rouge_scorer
# Insert the project root (biomed_extractor) into sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# load functions for import of clinicaltrials.gov data written previously
from app.data.loader import load_trials_json, extract_from_clinicaltrials
from app.nlp.pipelines import load_ner_pipeline_huggingface, load_ner_trained_pipeline
from app.nlp.utils import (
    compose_trial_text, chunk_text_by_chars, run_ner_on_long_text, clean_population_entities,
    merge_entities, extract_pico_from_merged_entities, normalize_intervention, is_substring_duplicate,
    deduplicate_intervention_entities, summarize_textRank, extract_comparator, remove_comparator_terms,
    clean_outcomes, process_trials_for_PICO, process_trials_for_retrained_PICO
)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction # for BLEU
import time



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


def evaluate_summary_rouge_score(df_gold, predicted_df, gold_col='short_description', pred_col='summary'):
    """
    Evaluate summaries using rouge_score and return mean precision, recall, F1 for ROUGE-1, ROUGE-2, ROUGE-L.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1_p, r1_r, r1_f = [], [], []
    r2_p, r2_r, r2_f = [], [], []
    rl_p, rl_r, rl_f = [], [], []
    
    refs = df_gold[gold_col].fillna("").astype(str).tolist()
    preds = predicted_df[pred_col].fillna("").astype(str).tolist()
    
    for ref, pred in zip(refs, preds):
        scores = scorer.score(ref, pred)
        r1_p.append(scores['rouge1'].precision); r1_r.append(scores['rouge1'].recall); r1_f.append(scores['rouge1'].fmeasure)
        r2_p.append(scores['rouge2'].precision); r2_r.append(scores['rouge2'].recall); r2_f.append(scores['rouge2'].fmeasure)
        rl_p.append(scores['rougeL'].precision); rl_r.append(scores['rougeL'].recall); rl_f.append(scores['rougeL'].fmeasure)

    results = {
        'SUMMARY_ROUGE-1': {'precision': sum(r1_p)/len(r1_p), 'recall': sum(r1_r)/len(r1_r), 'f1': sum(r1_f)/len(r1_f)},
        'SUMMARY_ROUGE-2': {'precision': sum(r2_p)/len(r2_p), 'recall': sum(r2_r)/len(r2_r), 'f1': sum(r2_f)/len(r2_f)},
        'SUMMARY_ROUGE-L': {'precision': sum(rl_p)/len(rl_p), 'recall': sum(rl_r)/len(rl_r), 'f1': sum(rl_f)/len(rl_f)},
    }
    return results

def add_per_row_rouge(df_gold, predicted_df, gold_col='short_description', pred_col='summary'):
    """
    Appends per-row ROUGE-1/2/L precision, recall, F1 columns to df_gold DataFrame.
    Returns the modified DataFrame and a ROUGE summary (averages).
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # Prepare per-row values lists
    for rouge_n in ['rouge1', 'rouge2', 'rougeL']:
        df_gold[f"{pred_col}_{rouge_n}_precision"] = 0.0
        df_gold[f"{pred_col}_{rouge_n}_recall"] = 0.0
        df_gold[f"{pred_col}_{rouge_n}_f1"] = 0.0
    # Iterate and calculate per-row
    refs = df_gold[gold_col].fillna("").astype(str).tolist()
    preds = predicted_df[pred_col].fillna("").astype(str).tolist()
    for idx, (ref, pred) in enumerate(zip(refs, preds)):
        scores = scorer.score(ref, pred)
        for rouge_n in ['rouge1', 'rouge2', 'rougeL']:
            df_gold.at[idx, f"{pred_col}_{rouge_n}_precision"] = scores[rouge_n].precision
            df_gold.at[idx, f"{pred_col}_{rouge_n}_recall"] = scores[rouge_n].recall
            df_gold.at[idx, f"{pred_col}_{rouge_n}_f1"] = scores[rouge_n].fmeasure
    # Optionally, compute means for summary
    rouge_summary = {
        f"{rouge_n}": {
            "precision": df_gold[f"{pred_col}_{rouge_n}_precision"].mean(),
            "recall":    df_gold[f"{pred_col}_{rouge_n}_recall"].mean(),
            "f1":        df_gold[f"{pred_col}_{rouge_n}_f1"].mean()
        }
        for rouge_n in ['rouge1', 'rouge2', 'rougeL']
    }
    return df_gold, rouge_summary

def evaluate_ner_model_partial_overlap(df_gold, ner_res_model, pico_cols,
                                       summary_gold_col='summary_reference',
                                       summary_pred_col='summary',
                                       add_rouge=True,
                                       processing_time=None,
                                       batch_throughput=None):
    """
    For each PICO column, compares gold and predicted values row-by-row
    using substring-overlap matching, and adds per-row precision, recall, and F1 columns
    to df_gold for detailed error analysis. Also adds per-row ROUGE and BLEU scores for summaries.
    Returns:
        tuple:
            - pd.DataFrame: df_gold with new columns for each evaluated field
            - pd.DataFrame: summary table with mean precision, recall, and F1 for each field
    """
    # Partial overlap for PICO columns
    for col in pico_cols:
        partial_precisions, partial_recalls, partial_f1s = [], [], []
        if col not in ner_res_model.columns:
            df_gold[f"{col}_partial_precision"] = None
            df_gold[f"{col}_partial_recall"] = None
            df_gold[f"{col}_partial_f1"] = None
            continue

        # Prepare an aligned list of predictions matching df_gold rows.
        # If lengths match, use direct column. If not, try to align by common ID columns
        # (e.g. df_gold['doc_id'] vs ner_res_model['nctId']). Otherwise pad or truncate.
        if len(ner_res_model) == len(df_gold):
            pred_iterable = ner_res_model[col].reset_index(drop=True).tolist()
        else:
            pred_iterable = None
            # try to align using common id columns
            if 'doc_id' in df_gold.columns and 'nctId' in ner_res_model.columns:
                try:
                    pred_series = ner_res_model.set_index('nctId')[col].reindex(df_gold['doc_id']).fillna('').tolist()
                    pred_iterable = pred_series
                except Exception:
                    pred_iterable = None
            # fallback: if same id column present in both
            if pred_iterable is None:
                for id_col in ['doc_id', 'nctId', 'nct_id', 'nctid']:
                    if id_col in df_gold.columns and id_col in ner_res_model.columns:
                        try:
                            pred_series = ner_res_model.set_index(id_col)[col].reindex(df_gold[id_col]).fillna('').tolist()
                            pred_iterable = pred_series
                            break
                        except Exception:
                            pred_iterable = None
            # final fallback: pad or truncate
            if pred_iterable is None:
                series = ner_res_model[col].reset_index(drop=True).astype(object).tolist()
                if len(series) < len(df_gold):
                    series = series + [''] * (len(df_gold) - len(series))
                else:
                    series = series[:len(df_gold)]
                pred_iterable = series

        for gold_cell, pred_cell in zip(df_gold[col], pred_iterable):
            gold_elems = elements_from_cell(gold_cell)
            pred_elems = elements_from_cell(pred_cell)
            p, r, f1 = substring_partial_overlap(gold_elems, pred_elems)
            partial_precisions.append(p)
            partial_recalls.append(r)
            partial_f1s.append(f1)

        # Assign lists (should now match df_gold length)
        df_gold[f"{col}_partial_precision"] = partial_precisions
        df_gold[f"{col}_partial_recall"] = partial_recalls
        df_gold[f"{col}_partial_f1"] = partial_f1s

    # Per-row ROUGE metrics for summary
    if add_rouge and summary_gold_col in df_gold.columns and summary_pred_col in ner_res_model.columns:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        smoothie = SmoothingFunction().method4
        bleu_scores = []
        # Initialize columns
        for rn in ['rouge1', 'rouge2', 'rougeL']:
            df_gold[f"{summary_pred_col}_{rn}_precision"] = 0.0
            df_gold[f"{summary_pred_col}_{rn}_recall"] = 0.0
            df_gold[f"{summary_pred_col}_{rn}_f1"] = 0.0
        # Compute per-row scores
        refs = df_gold[summary_gold_col].fillna("").astype(str).tolist()
        # Align predictions for summaries similar to PICO alignment above
        if len(ner_res_model) == len(df_gold):
            preds = ner_res_model[summary_pred_col].fillna("").astype(str).tolist()
        else:
            preds = None
            if 'doc_id' in df_gold.columns and 'nctId' in ner_res_model.columns:
                try:
                    preds = ner_res_model.set_index('nctId')[summary_pred_col].reindex(df_gold['doc_id']).fillna("").astype(str).tolist()
                except Exception:
                    preds = None
            if preds is None:
                for id_col in ['doc_id', 'nctId', 'nct_id', 'nctid']:
                    if id_col in df_gold.columns and id_col in ner_res_model.columns:
                        try:
                            preds = ner_res_model.set_index(id_col)[summary_pred_col].reindex(df_gold[id_col]).fillna("").astype(str).tolist()
                            break
                        except Exception:
                            preds = None
            if preds is None:
                series = ner_res_model[summary_pred_col].reset_index(drop=True).astype(str).tolist()
                if len(series) < len(df_gold):
                    series = series + [""] * (len(df_gold) - len(series))
                else:
                    series = series[:len(df_gold)]
                preds = series

        for idx, (ref, pred) in enumerate(zip(refs, preds)):
            scores = scorer.score(ref, pred)
            for rn in ['rouge1', 'rouge2', 'rougeL']:
                df_gold.at[idx, f"{summary_pred_col}_{rn}_precision"] = scores[rn].precision
                df_gold.at[idx, f"{summary_pred_col}_{rn}_recall"] = scores[rn].recall
                df_gold.at[idx, f"{summary_pred_col}_{rn}_f1"] = scores[rn].fmeasure
            # BLEU
            ref_tokens = [ref.split()]
            pred_tokens = pred.split()
            bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
            bleu_scores.append(bleu)
        df_gold[f"{summary_pred_col}_bleu"] = bleu_scores

    # Calculate proportion of documents with all PICO elements partially correct
    all_partial_correct = []
    for idx, row in df_gold.iterrows():
        is_correct = True
        for col in pico_cols:
            # Use partial recall as indicator (can adjust threshold if needed)
            recall = row.get(f"{col}_partial_recall", 0.0)
            # Consider "correct" if recall == 1.0 (all gold standard items are retrieved)
            if recall < 1.0:
                is_correct = False
                break
        all_partial_correct.append(is_correct)
    proportion_partial_correct = sum(all_partial_correct) / len(all_partial_correct) if all_partial_correct else 0.0

    # Summary table
    rows = []
    # Add PICO metrics summary
    for col in pico_cols:
        rows.append({
            "element": col.replace("_extracted", ""),
            "precision": df_gold[f"{col}_partial_precision"].mean(),
            "recall":    df_gold[f"{col}_partial_recall"].mean(),
            "f1":        df_gold[f"{col}_partial_f1"].mean()
        })
    # Add ROUGE metrics summary if available
    if add_rouge:
        for rn, rn_name in zip(['rouge1', 'rouge2', 'rougeL'], ["SUMMARY_ROUGE-1", "SUMMARY_ROUGE-2", "SUMMARY_ROUGE-L"]):
            rows.append({
                "element": rn_name,
                "precision": df_gold[f"{summary_pred_col}_{rn}_precision"].mean(),
                "recall":    df_gold[f"{summary_pred_col}_{rn}_recall"].mean(),
                "f1":        df_gold[f"{summary_pred_col}_{rn}_f1"].mean()
            })
        # Add BLEU score summary
        rows.append({
            "element": "SUMMARY_BLEU",
            "precision": None,
            "recall": None,
            "f1": None,
            "bleu": sum(bleu_scores) / len(bleu_scores)
        })
    
    # Add proportion of fully partially correct rows
    rows.append({
        "element": "PROPORTION_ALL_PICO_PARTIAL_CORRECT",
        "precision": None,
        "recall": None,
        "f1": None,
        "bleu": None,
        "proportion": proportion_partial_correct
    })

    # Add processing time and throughput if provided
    if processing_time is not None and batch_throughput is not None:
        rows.append({
            "element": "PROCESSING_TIME",
            "precision": None,
            "recall": None,
            "f1": None,
            "bleu": None,
            "proportion": None,
            "total_seconds": processing_time[0],
            "avg_seconds_per_doc": processing_time[1],
            "median_seconds_per_doc": processing_time[2]
        })
        rows.append({
            "element": "BATCH_THROUGHPUT",
            "precision": None,
            "recall": None,
            "f1": None,
            "bleu": None,
            "proportion": None,
            "docs_per_minute": batch_throughput
        })
    # Format summary table
    summary_table = pd.DataFrame(rows).set_index("element")

    return df_gold, summary_table


def proportion_all_pico_correct(df_gold, ner_res_model, pico_cols, id_col='doc_id'):
    """
    Returns the proportion of documents for which all PICO elements are perfectly matched.
    """
    correct = []
    for idx, row in df_gold.iterrows():
        all_correct = True
        for col in pico_cols:
            gold_elems = elements_from_cell(row[col])
            pred_elems = elements_from_cell(ner_res_model.loc[idx, col])
            # Require exact match for all elements
            if gold_elems != pred_elems:
                all_correct = False
                break
        correct.append(all_correct)
    proportion = sum(correct) / len(correct) if correct else 0.0
    return proportion

def measure_processing_time(process_func, *args, **kwargs):
    """
    Measures the time taken to process a batch of documents.
    Returns total time, average time per document, and median time per document.
    """
    start = time.time()
    result = process_func(*args, **kwargs)
    end = time.time()
    total_time = end - start
    num_docs = len(args[0]) if hasattr(args[0], '__len__') else 1
    avg_time = total_time / num_docs if num_docs else 0
    # For median, measure per-row if possible
    if hasattr(result, '__len__') and num_docs > 1:
        per_row_times = []
        for i in range(num_docs):
            row_start = time.time()
            _ = process_func(args[0].iloc[[i]], *args[1:], **kwargs)
            row_end = time.time()
            per_row_times.append(row_end - row_start)
        median_time = sorted(per_row_times)[num_docs // 2]
    else:
        median_time = avg_time
    return total_time, avg_time, median_time

def calculate_batch_throughput(total_time, num_docs):
    """
    Calculates batch throughput: number of documents processed per minute.
    """
    if total_time == 0:
        return float('inf')
    throughput = num_docs / (total_time / 60)
    return throughput

if __name__ == "__main__":
    # Get the PROJECT ROOT (biomed-extractor/)
    PROJECT_ROOT = os.path.expanduser('~/Documents/github/biomed_extractor')
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
    #mymodel = "kamalkraj/BioELECTRA-PICO"
    #ner_pipeline = load_ner_pipeline_huggingface(mymodel)
    #ner_res_model = process_trials_for_PICO(mydf_manual_annotation, ner_pipeline)
    # for self-trained model
    #mymodel = "app/model/nlpie_compact_biobert_PICO"
    #mymodel = "app/model/nlpie_bio-mobilebert_PICO"
    mymodel = "app/model/dmis-lab_biobert-v1.1"
    ner_pipeline = load_ner_trained_pipeline(mymodel)
    ner_res_model = process_trials_for_retrained_PICO(mydf_manual_annotation, ner_pipeline)

    ner_res_model.sort_values(by=['nctId'], inplace=True)
    # rename columns to match gold standard
    ner_res_model.rename(columns={'population_extracted': 'population',
                                'intervention_extracted': 'intervention',
                                'comparator_extracted': 'comparator',
                                'outcome_extracted': 'outcome',
                                'summary_extracted': 'summary'}, inplace=True)
    
    # evaluate results
    pico_cols = ["population", "intervention", "comparator", "outcome"]
    # Measure processing time and throughput for extraction
    total_time_load, avg_time_load, median_time_load = measure_processing_time(load_ner_trained_pipeline, mymodel)
    print("loading model time: ", total_time_load)
    total_time, avg_time, median_time = measure_processing_time(process_trials_for_retrained_PICO, mydf_manual_annotation, ner_pipeline)
    throughput = calculate_batch_throughput(total_time, len(mydf_manual_annotation))
    #df_gold_with_partial, evaluation_table = evaluate_ner_model_partial_overlap(df_gold.copy(), ner_res_model, pico_cols)
    df_gold_with_partial, evaluation_table = evaluate_ner_model_partial_overlap(df_gold.copy(), 
                                                                                ner_res_model, 
                                                                                pico_cols, 
                                                                                summary_gold_col='summary', 
                                                                                summary_pred_col='summary', 
                                                                                add_rouge=True,
                                                                                processing_time=(total_time, avg_time, median_time),
                                                                                batch_throughput=throughput)
    print(evaluation_table)
    print(df_gold_with_partial.head())
#    df_gold_with_partial.to_csv("data/results_BioELECTRA.csv", index=False)
#    evaluation_table.to_csv("data/results_BioELECTRA_summary.csv", index=True)
   # df_gold_with_partial.to_csv("data/results_compact_biobert_PICO.csv", index=False)
   # evaluation_table.to_csv("data/results_compact_biobert_PICO_summary.csv", index=True)
   # df_gold_with_partial.to_csv("data/results_bio-mobilebert_PICO.csv", index=False)
   # evaluation_table.to_csv("data/results_bio-mobilebert_PICO_summary.csv", index=True)
    df_gold_with_partial.to_csv("data/results_biobert_PICO.csv", index=False)
    evaluation_table.to_csv("data/results_biobert_PICO_summary.csv", index=True)
    prop_all_pico = proportion_all_pico_correct(df_gold, ner_res_model, pico_cols, id_col='doc_id')
    print(f"Proportion of documents with all PICO elements correct: {prop_all_pico:.3f}")
  
    
