import pytest
import pandas as pd
from app.data.loader import (
    flatten_list_of_dicts,
    clean_text,
    split_eligibility,
    extract_from_clinicaltrial,
    extract_from_clinicaltrials,
    load_trials_csv
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


def test_extract_from_clinicaltrial():
    mock_df = pd.DataFrame({
        'protocolSection': [{
            'nctId': 'NCT00000002',
            'briefSummary': 'Brief summary',
            'detailedDescription': 'Detailed description',
            'eligibilityCriteria': 'Inclusion Criteria: test. Exclusion Criteria: test.',
            'interventions': [{'type': 'DRUG', 'name': 'DrugY 100mg', 'description': 'Test drug'}],
            'primaryOutcomes': [{'measure': 'Memory score', 'description': 'Test', 'timeFrame': '6 months'}]
        }]
    })

    result = extract_from_clinicaltrial(mock_df)
    assert result.shape[0] == 1
    assert 'nctid' in result.columns[0].lower()
    assert 'drugy' in result['intervention_name_clean'].iloc[0].lower()
    assert 'memory score' in result['outcomes_name'].iloc[0].lower()
    assert 'test' in result['inclusion_criteria'].iloc[0].lower()

def test_extract_from_clinicaltrials():
    mock_df = pd.DataFrame({
        'protocolSection': [{
            'identificationModule': {'nctId': 'NCT00000003'},
            'descriptionModule': {
                'briefSummary': 'Brief summary for trial',
                'detailedDescription': 'Detailed description for trial'
            },
            'eligibilityModule': {
                'eligibilityCriteria': 'Inclusion Criteria: must be healthy. Exclusion Criteria: allergic to drug.'
            },
            'armsInterventionsModule': {
                'interventions': [{'type': 'DRUG', 'name': 'DrugZ 200mg', 'description': 'Test drug Z'}]
            },
            'outcomesModule': {
                'primaryOutcomes': [{'measure': 'Reaction time', 'description': 'Test', 'timeFrame': '3 months'}]
            }
        }]
    })

    result = extract_from_clinicaltrials(mock_df)

    assert result.shape[0] == 1
    assert 'nctid' in result.columns[0].lower()
    assert 'drugz' in result['intervention_name_clean'].iloc[0].lower()
    assert 'reaction time' in result['outcomes_name'].iloc[0].lower()
    assert 'must be healthy' in result['inclusion_criteria'].iloc[0].lower()
    assert 'allergic to drug' in result['exclusion_criteria'].iloc[0].lower()


def test_load_trials_csv_with_column_standardization(tmp_path):
    # Create a mock CSV file with expected columns
    test_file = tmp_path / "example_trials.csv"
    test_file.write_text(
        "NCT Number,Brief Summary,Conditions,Interventions,Primary Outcome Measures\n"
        "NCT00000001,Test summary,ConditionX,DrugX,OutcomeX"
    )

    # Define a mock version of DATA_DIR pointing to tmp_path
    df = load_trials_csv(filepath=tmp_path, filename="example_trials.csv")

    # Check that the DataFrame has the standardized columns
    expected_columns = [
        'nctId',
        'briefSummary',
        'intervention_name_clean',
        'outcomes_name',
        'detailedDescription',
        'inclusion_criteria',
        'exclusion_criteria'
    ]
    for col in expected_columns:
        assert col in df.columns

    # Check that the values are correctly preserved
    assert df.loc[0, 'nctId'] == 'NCT00000001'
    assert df.loc[0, 'briefSummary'] == 'Test summary'
    assert df.loc[0, 'intervention_name_clean'] == 'DrugX'
    assert df.loc[0, 'outcomes_name'] == 'OutcomeX'
