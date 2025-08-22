import pytest
import pandas as pd
from app.nlp.utils import (
    compose_trial_text, chunk_text_by_chars, run_ner_on_long_text, clean_population_entities,
    merge_entities, extract_pico_from_merged_entities, normalize_intervention, is_substring_duplicate,
    deduplicate_intervention_entities, summarize_textRank, extract_comparator, remove_comparator_terms,
    clean_outcomes, process_trials_for_PICO
)

# Mock NER pipeline (returns fake entities for a given chunk)
class MockNer:
    def __call__(self, text):
        # Returns example entities for tests
        # This will split text into word entities just for testing
        return [
            {'entity_group': 'participants', 'word': word, 'start': i, 'end': i+len(word)}
            for i, word in enumerate(text.split())
        ]

def test_compose_trial_text_basic():
    row = {'briefSummary': 'Short summary', 'detailedDescription': 'Detailed desc'}
    assert compose_trial_text(row) == 'Short summary Detailed desc'
    row2 = {'briefSummary': 'Just summary', 'detailedDescription': None}
    assert compose_trial_text(row2) == 'Just summary'
    row3 = {'briefSummary': 'Just summary', 'detailedDescription': ''}
    assert compose_trial_text(row3) == 'Just summary'

def test_chunk_text_by_chars():
    text = 'a' * 50
    chunks = list(chunk_text_by_chars(text, chunk_char_length=20, overlap=5))
    assert len(chunks) > 2
    # Check overlap
    assert chunks[1][:5] == chunks[0][-5:]

def test_run_ner_on_long_text():
    ner = MockNer()
    text = "John is a subject"
    result = run_ner_on_long_text(text, ner, chunk_char_length=5, overlap=2)
    # Should build entities for each chunk and merge them
    assert isinstance(result, list)
    assert all('entity_group' in e for e in result)

def test_clean_population_entities():
    ents = ["Young adults", "patients with condition", "foo", "male", "male"]
    result = clean_population_entities(ents)
    entities = [e.strip() for e in result.split(';')]
    # Only demographic/diagnostic words appear, no duplicates, no "foo"
    assert "foo" not in entities
    assert "male" in entities
    assert entities.count("male") == 1

def test_merge_entities():
    # Two adjacent 'participants' entities, then a space
    entities = [
        {'entity_group': 'participants', 'word': 'young', 'start': 0, 'end': 5},
        {'entity_group': 'participants', 'word': 'adults', 'start': 5, 'end': 11},
        {'entity_group': 'intervention', 'word': 'drug', 'start': 12, 'end': 16}
    ]
    merged = merge_entities(entities)
    assert any(m['word'] == 'young adults' for m in merged)
    assert len(merged) == 2

def test_extract_pico_from_merged_entities():
    entities = [
        {'entity_group': 'participants', 'word': 'patients', 'start': 0, 'end': 8},
        {'entity_group': 'intervention', 'word': 'drugA', 'start': 9, 'end': 14},
        {'entity_group': 'outcome', 'word': 'improved', 'start': 15, 'end': 23}
    ]
    pico = extract_pico_from_merged_entities(entities)
    assert pico['participants'] == 'patients'
    assert pico['intervention'] == 'drugA'
    assert pico['outcome'] == 'improved'

def test_normalize_intervention():
    s = "DrugA (oral) or DrugB and DrugC, 10mg."
    norm = normalize_intervention(s)
    assert "or" not in norm and "and" not in norm
    assert "(" not in norm and "," not in norm

def test_is_substring_duplicate():
    deduped = ['drug treatment', 'placebo']
    assert is_substring_duplicate('drug', deduped)
    assert is_substring_duplicate('treatment', deduped)
    assert not is_substring_duplicate('random', deduped)

def test_deduplicate_intervention_entities():
    ents = ['DrugA', 'DrugA', 'drug a', 'DrugX', 'DrugX + DrugA']
    result = deduplicate_intervention_entities(ents)
    # DrugA and DrugX should each appear once, DrugA subset of DrugX+DrugA
    assert 'druga' in result
    assert result.count('druga') == 1

def test_summarize_textRank():
    text = "This is one sentence. This is another. And here is the third."
    summary = summarize_textRank(text)
    assert isinstance(summary, str)
    assert len(summary.split('.')) <= 3  # Two sentences

def test_extract_comparator():
    interventions = "DrugA; Placebo; DrugB"
    comp = extract_comparator(interventions)
    assert comp == 'placebo'
    assert extract_comparator("none") == ''

def test_remove_comparator_terms():
    interventions = "DrugA; Placebo; DrugB"
    cleaned = remove_comparator_terms(interventions)
    assert "placebo" not in cleaned
    assert "DrugA" in cleaned

def test_clean_outcomes():
    outcomes = "Improved memory; [Reduced anxiety]; *Side effects*; ;"
    cleaned = clean_outcomes(outcomes)
    assert "improved memory" in cleaned
    assert "side effects" in cleaned
    assert cleaned.count(";") == 2

def test_process_trials_for_PICO():
    # Build mock DataFrame for trials
    df = pd.DataFrame([{
        'briefSummary': 'Test summary',
        'detailedDescription': 'More info on DrugA and DrugB.',
        'inclusion_criteria': 'men; adults; patients',
        # other fields can be empty/mocked
    }])
    ner = MockNer()
    out = process_trials_for_PICO(df, ner)
    # Outputs added columns
    assert "population_extracted" in out.columns
    assert "intervention_extracted" in out.columns
    assert "outcome_extracted" in out.columns
    assert "summary_extracted" in out.columns
    assert "comparator_extracted" in out.columns
