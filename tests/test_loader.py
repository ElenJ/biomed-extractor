import pytest
import pandas as pd
from app.data.loader import (
    flatten_list_of_dicts,
    clean_text,
    split_eligibility,
    extract_elegibility_trtm_from_clinicaltrials
)

# --- flatten_list_of_dicts ---
def test_flatten_list_of_dicts():
    input_data = [{'type': 'DRUG', 'name': 'DrugX', 'description': 'Test drug'}]
    keys = ['type', 'name', 'description']
    result = flatten_list_of_dicts(input_data, keys)
    assert result == ['type: DRUG; name: DrugX; description: Test drug']

# --- clean_text ---
def test_clean_text_removes_trademarks_and_html():
    raw = "This is a test® with <b>HTML</b> and ™ symbols * and 50mg."
    cleaned = clean_text(raw)
    assert '®' not in cleaned
    assert '™' not in cleaned
    assert '<b>' not in cleaned
    assert '*' not in cleaned
    assert '50mg' in cleaned  # dosage removal is handled elsewhere
    assert cleaned == cleaned.lower()

# --- split_eligibility ---
def test_split_eligibility_both_present():
    text = "Inclusion Criteria: must be 18+. Exclusion Criteria: allergy to drug."
    inc, exc = split_eligibility(text)
    assert "must be 18+" in inc.lower()
    assert "allergy to drug" in exc.lower()

def test_split_eligibility_only_inclusion():
    text = "Inclusion Criteria: must be 18+."
    inc, exc = split_eligibility(text)
    assert "must be 18+" in inc.lower()
    assert exc == ''

def test_split_eligibility_only_exclusion():
    text = "Exclusion Criteria: allergy to drug."
    inc, exc = split_eligibility(text)
    assert inc == ''
    assert "allergy to drug" in exc.lower()

# --- extract_elegibility_trtm_from_clinicaltrials ---
def test_extract_elegibility_trtm_from_clinicaltrials():
    mock_df = pd.DataFrame({
        'protocolSection': [{
            'identificationModule': {'nctId': 'NCT00000001'},
            'descriptionModule': {'briefSummary': 'Test summary'},
            'eligibilityModule': {'eligibilityCriteria': 'Inclusion Criteria: test. Exclusion Criteria: test.'},
            'armsInterventionsModule': {
                'interventions': [{'type': 'DRUG', 'name': 'DrugX 50mg', 'description': 'Test drug'}]
            },
            'outcomesModule': {
                'primaryOutcomes': [{'measure': 'Cognitive score', 'description': 'Test', 'timeFrame': '12 weeks'}]
            }
        }]
    })

    result = extract_elegibility_trtm_from_clinicaltrials(mock_df)
    assert result.shape[0] == 1
    assert 'nctid' in result.columns[0].lower()
    assert 'drugx' in result['intervention_name_clean'].iloc[0].lower()
    assert 'cognitive score' in result['outcomes_name'].iloc[0].lower()
    assert 'test' in result['inclusion_criteria'].iloc[0].lower()
