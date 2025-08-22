import pandas as pd
# extractive summarization with textrank:
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

import re
import pandas as pd
from difflib import SequenceMatcher
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

GENERIC_WORDS = {'medications', 'treatment', 'antipsychotics', 'antidepressants'}

def compose_trial_text(row):
    """
    Combine the key trial description fields ("briefSummary" and "detailedDescription")
    into a single text string.

    Args:
        row (pd.Series or dict): A row of a DataFrame representing a clinical trial.
            Must contain the keys 'briefSummary' and optionally 'detailedDescription'.

    Returns:
        str: The combined text, with the detailed description appended if present.
    """
    text = row['briefSummary']
    if pd.notna(row.get('detailedDescription')) and row['detailedDescription'].strip():
        text += " " + row['detailedDescription']
    return text


def chunk_text_by_chars(text, chunk_char_length=1200, overlap=100):
    """
    Split a long text into (potentially overlapping) chunks by character count.

    Args:
        text (str): The input text to be chunked.
        chunk_char_length (int): Maximum number of characters per chunk.
        overlap (int): Number of characters chunks should overlap.

    Yields:
        str: The next text chunk.
    """
    start = 0
    n_chars = len(text)
    while start < n_chars:
        end = min(start + chunk_char_length, n_chars)
        yield text[start:end]
        if end == n_chars:
            break
        start = end - overlap  # introduce overlap for better span boundary coverage


def run_ner_on_long_text(text, ner_pipeline, chunk_char_length=1200, overlap=100):
    """
    Run a NER pipeline on a (potentially long) text, by processing it in chunks and aggregating results.

    Args:
        text (str): The input text to process.
        ner_pipeline (callable): A HuggingFace transformers NER pipeline (or compatible).
        chunk_char_length (int): Chunk length in characters.
        overlap (int): Overlap in characters between chunks.

    Returns:
        list: List of all entity dicts found in text (across all chunks).
    """
    entities = []
    for chunk in chunk_text_by_chars(text, chunk_char_length, overlap):
        ents = ner_pipeline(chunk)
        entities.extend(ents)
    return entities


def clean_population_entities(pop_entities):
    """
    Clean and deduplicate extracted population entity strings.

    Populations must either contain a known demographic/diagnosis keyword,
    or be at least 8 characters long. Exact duplicates are removed.

    Args:
        pop_entities (list of str): List of population entity strings.

    Returns:
        str: Cleaned, deduplicated, semicolon-separated population entities.
    """
    min_len = 8
    demo_kw = [
        'patients', 'subjects', 'adults', 'children', 'individuals', 
        'men', 'women', 'male', 'female', 'participants'
    ]
    result = []
    for ent in pop_entities:
        ent_clean = ent.strip().strip(';,')
        # Keep always if keyword found; else only if long enough
        if any(kw in ent_clean.lower() for kw in demo_kw):
            result.append(ent_clean)
        elif len(ent_clean) >= min_len:
            result.append(ent_clean)
    return '; '.join(sorted(set(result)))


def merge_entities(entities):
    """
    Merge consecutive tokens of the same entity group (if they are adjacent) into a single phrase.

    Args:
        entities (list of dict): Each dict must have keys 'entity_group', 'word', 'start', 'end'.

    Returns:
        list of dict: Each dict describes a merged entity with same keys as input.
    """
    if not entities:
        return []
    # Sort entities by start character position
    entities = sorted(entities, key=lambda x: x['start'])
    merged = []
    current = None
    for ent in entities:
        # Start a new merged entity unless it's of the same group and directly adjacent
        if (current is None or
            ent['entity_group'] != current['entity_group'] or
            ent['start'] != current['end']):
            if current:
                merged.append(current)
            current = {
                'entity_group': ent['entity_group'],
                'word': ent['word'],
                'start': ent['start'],
                'end': ent['end']
            }
        else:
            # Continue current merged entity
            current['word'] += ' ' + ent['word']
            current['end'] = ent['end']
    if current:
        merged.append(current)
    return merged


def extract_pico_from_merged_entities(entities):
    """
    Group entities by PICO type and collect their text.

    Args:
        entities (list of dict): List of merged entities. Each dict must have keys:
            'entity_group' and 'word'.

    Returns:
        dict: For each PICO group ("participants", "intervention", "comparator", "outcome"),
            a semicolon-separated string of unique phrases.
    """
    pico_dict = {"participants": [], "intervention": [], "comparator": [], "outcome": []}
    for ent in entities:
        group = ent['entity_group'].lower()
        if group in pico_dict:
            pico_dict[group].append(ent['word'])
    # Deduplicate and join phrases per group
    return {k: "; ".join(sorted(set(v))) if v else "" for k, v in pico_dict.items()}


def normalize_intervention(ent):
    """
    Normalize an intervention phrase for comparison/consistency:
    lowercase, remove punctuation, brackets, redundant words, etc.

    Args:
        ent (str): Intervention phrase.

    Returns:
        str: Normalized intervention phrase.
    """
    ent = ent.lower().strip()
    ent = re.sub(r"\(([^\)]*)\)", "", ent)
    ent = re.sub(r"[^\w\s]", " ", ent)
    ent = re.sub(r"\s+", " ", ent)
    ent = re.sub(r"\bor\b.*$", "", ent)
    ent = re.sub(r"\band\b.*$", "", ent)
    ent = ent.strip()
    return ent


def is_substring_duplicate(e, deduped):
    """
    Check if a string is a substring of, or contains a substring from, any string in a given list.

    Args:
        e (str): Candidate string to check.
        deduped (list of str): List of already included entities.

    Returns:
        bool: True if e is a (or contains a) substring of any item in deduped (excluding exact match).
    """
    for existing in deduped:
        if e != existing and (e in existing or existing in e):
            return True
    return False


def deduplicate_intervention_entities(entities, threshold=0.85):
    """
    Deduplicate and normalize intervention entities using substring and fuzzy similarity.

    Args:
        entities (list of str): Raw intervention phrases.
        threshold (float): Similarity threshold for fuzzy deduplication.

    Returns:
        str: Clean, deduplicated, semicolon-separated interventions string.
    """
    cleaned = [normalize_intervention(e) for e in entities if len(e.strip()) > 2]
    cleaned = [ent for ent in cleaned if ent and ent not in GENERIC_WORDS]
    final = []
    for cand in cleaned:
        if not is_substring_duplicate(cand, final) and \
           not any(SequenceMatcher(None, cand, d).ratio() > threshold for d in final):
            final.append(cand)
    return "; ".join(sorted(final))


def summarize_textRank(text):
    """
    Create an extractive 2-sentence summary of the input using the TextRank algorithm.

    Args:
        text (str): The input text.

    Returns:
        str: The summary as a single string (2 sentences concatenated).
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences_count=2)
    return ' '.join(str(sentence) for sentence in summary)


def extract_comparator(interventions):
    """
    Identify if a comparator keyword is present in an interventions string.

    Args:
        interventions (str): A string containing extracted interventions.

    Returns:
        str: The first recognized comparator present, or '' if none are found.
    """
    if not isinstance(interventions, str):
        return ''
    comparators = [
        'placebo', 'standard care', 'usual care', 'sham', 'control',
        'standard therapy', 'vehicle', 'observation only', 'waitlist',
        'best supportive care'
    ]
    interventions_lower = interventions.lower()
    for comp in comparators:
        if comp in interventions_lower:
            return comp
    return ''


def remove_comparator_terms(interventions):
    """
    Remove any known comparator terms from an interventions string.

    Args:
        interventions (str): A semicolon-delimited list of intervention strings.

    Returns:
        str: Semicolon-separated string without comparator items.
    """
    if not isinstance(interventions, str):
        return interventions
    comparators = [
        'placebo', 'standard care', 'usual care', 'sham', 'control',
        'standard therapy', 'vehicle', 'observation only', 'waitlist',
        'best supportive care'
    ]
    parts = [p.strip() for p in interventions.split(';')]
    cleaned = [p for p in parts if p and p.lower() not in comparators]
    return '; '.join(cleaned)


def clean_outcomes(outcome_str):
    """
    Clean and deduplicate the string representation of outcome text.

    Args:
        outcome_str (str): Raw outcome information.

    Returns:
        str: Cleaned, deduplicated, semicolon-separated outcomes as a string.
    """
    if not isinstance(outcome_str, str):
        return ''
    # Remove various punctuation/special chars
    cleaned = re.sub(r'[\[\]()*\\/:#]', '', outcome_str)
    cleaned = re.sub(r'\s+', ' ', cleaned)   # Replace multiple spaces
    cleaned = cleaned.replace(' ,', ',').replace(' ;', ';')
    cleaned = cleaned.lower().strip()
    # Split on semicolon, deduplicate
    outcomes = [o.strip(' .;,-') for o in cleaned.split(';')]
    outcomes = list(dict.fromkeys([o for o in outcomes if o and len(o) > 1]))
    return '; '.join(outcomes)


def process_trials_for_PICO(df, ner_pipeline):
    """
    Main function to extract and clean PICO elements and summaries from a DataFrame of clinical trials.

    Applies NER, merging, extraction, and cleaning for each record.
    Adds new columns to the DataFrame with extracted/cleaned PICO elements.

    Args:
        df (pd.DataFrame): DataFrame containing trial records ('briefSummary', 'detailedDescription', 'inclusion_criteria').
        ner_pipeline (callable): NER pipeline for entity extraction.

    Returns:
        pd.DataFrame: The input DataFrame with new columns:
            ['population_extracted', 'intervention_extracted', 
            'outcome_extracted', 'comparator_extracted', 'summary_extracted'].
    """
    pop, intervention, outcome, summary = [], [], [], []
    for _, row in df.iterrows():
        main_text = compose_trial_text(row)
        inclusion_text = row.get("inclusion_criteria", "")
        # Apply NER and merge for main and inclusion text
        main_entities = merge_entities(run_ner_on_long_text(main_text, ner_pipeline))
        inclusion_entities = merge_entities(run_ner_on_long_text(str(inclusion_text), ner_pipeline))
        pico_main = extract_pico_from_merged_entities(main_entities)
        pico_inc = extract_pico_from_merged_entities(inclusion_entities)
        # Population: use inclusion, Intervention/Outcome: use main text
        cleaned_population = clean_population_entities(pico_inc["participants"].split(";")) if pico_inc["participants"] else ""
        pop.append(cleaned_population)
        raw_interventions = pico_main["intervention"].split(";") if pico_main["intervention"] else []
        cleaned_intervention = deduplicate_intervention_entities(raw_interventions)
        intervention.append(cleaned_intervention)
        outcome.append(pico_main["outcome"])
        summary2sent = summarize_textRank(main_text)
        summary.append(summary2sent)
    df["population_extracted"] = pop
    df["intervention_extracted"] = intervention
    df["outcome_extracted"] = outcome

    # Final post-processing/cleaning
    df['outcome_extracted'] = df['outcome_extracted'].apply(clean_outcomes)
    df['comparator_extracted'] = df['intervention_extracted'].apply(extract_comparator)
    df['intervention_extracted'] = df['intervention_extracted'].apply(remove_comparator_terms)
    df["summary_extracted"] = summary
    return df