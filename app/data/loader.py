import os
import pandas as pd
import re
import unicodedata

# Get the PROJECT ROOT (biomed-extractor/)
PROJECT_ROOT = 'c:\\Users\\USER\\Documents\\github\\biomed_extractor'

# Data directory at top level
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def load_trials_csv(filepath, filename):
    path = os.path.join(filepath, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} records from {filename}")
    process_df = extract_from_clinicaltrials_csv(df)
    return process_df

def extract_from_clinicaltrials_csv(df):
    
    df_csv_focused = df[['NCT Number', 'Brief Summary','Interventions', 'Primary Outcome Measures']].copy()
    required = ['detailedDescription', 'inclusion_criteria', 'exclusion_criteria']
    df_csv_focused = ensure_columns(df_csv_focused, required)

    standardized_columns = {
    'NCT Number': 'nctId',
    'Brief Summary': 'briefSummary',
    'descriptionModule.detailedDescription': 'detailedDescription',
    'inclusion_criteria': 'inclusion_criteria',
    'exclusion_criteria': 'exclusion_criteria',
    'Interventions': 'intervention_name_clean',
    'Primary Outcome Measures': 'outcomes_name'
    }

    df_csv_focused = df_csv_focused.rename(columns=standardized_columns)
    
    desired_order = [
        'nctId',
        'briefSummary',
        'detailedDescription',
        'inclusion_criteria',
        'exclusion_criteria',
        'intervention_name_clean',
        'outcomes_name'
    ]

    df_csv_focused = df_csv_focused[desired_order]

    return df_csv_focused

def load_trials_json(filepath, filename):
    path = os.path.join(filepath, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_json(path)
    print(f"Loaded {len(df)} records from {filename}")
    return df

# From here: functions to process json files
# Helper function to flatten list of dictionaries
def flatten_list_of_dicts(lst, keys):
    if isinstance(lst, list):
        return ['; '.join(f"{k}: {d.get(k, '')}" for k in keys if isinstance(d, dict)) for d in lst]
    return lst

# Define a function to clean the text
def clean_text(text):
    if pd.isna(text):
        return ''
    # Normalize encoding
    text = unicodedata.normalize('NFKD', text)
    # Remove HTML/XML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove non-printable characters
    text = ''.join(c for c in text if c.isprintable()) 
    # Remove asterisks
    text = text.replace('*', '; ')
    # Remove asterisks at beginning of sentences
    text = text.replace(':;', ':')
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # drop trademark symbols
    text = re.sub(r'[\u00AE\u2122\u2120]', '', text)
    # Convert to lowercase
    text = text.lower() 
    return text

# Function to split eligibility into inclusion and exclusion criteria
def split_eligibility(text):
    inclusion = ''
    exclusion = ''
    if isinstance(text, str):
        text_lower = text.lower()
        inc_start = text_lower.find('inclusion criteria:')
        exc_start = text_lower.find('exclusion criteria:')
        if inc_start != -1 and exc_start != -1:
            inclusion = text[inc_start + len('inclusion criteria:'):exc_start].strip()
            exclusion = text[exc_start + len('exclusion criteria:'):].strip()
        elif inc_start != -1:
            inclusion = text[inc_start + len('inclusion criteria:'):].strip()
        elif exc_start != -1:
            exclusion = text[exc_start + len('exclusion criteria:'):].strip()
    return pd.Series([inclusion, exclusion])




def flatten_and_extract(df, intervention_col, outcome_col):
    df['interventions_clean'] = df[intervention_col].apply(
        lambda x: flatten_list_of_dicts(x, ['type', 'name', 'description'])
    )
    df['primaryOutcomes_clean'] = df[outcome_col].apply(
        lambda x: flatten_list_of_dicts(x, ['measure', 'description', 'timeFrame'])
    )
    df = df.explode('interventions_clean').explode('primaryOutcomes_clean').reset_index(drop=True)
    df['intervention_name'] = df['interventions_clean'].str.extract(r'name:\s*(.*?)\s*;?\s*description:')
    df['outcomes_name'] = df['primaryOutcomes_clean'].str.extract(r'measure:\s*(.*?)\s*;?\s*description:')
    df['intervention_name_clean'] = df['intervention_name'].str.replace(r'\s*\d+\s*(mg|mcg|g|ml)', '', case=False, regex=True)
    return df

def ensure_columns(df, required_columns):
    for col in required_columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df

def aggregate_and_clean(df, id_col, summary_col, desc_col, elig_col):
    combined_df = df.groupby(id_col).agg({
        summary_col: lambda x: '; '.join(pd.unique(x.dropna())),
        desc_col: lambda x: '; '.join(pd.unique(x.dropna())),
        elig_col: lambda x: '; '.join(pd.unique(x.dropna())),
        'intervention_name_clean': lambda x: '; '.join(pd.unique(x.dropna())),
        'outcomes_name': lambda x: '; '.join(pd.unique(x.dropna()))
    }).reset_index()

    combined_df['eligibility_clean'] = combined_df[elig_col].apply(clean_text)
    combined_df[['inclusion_criteria', 'exclusion_criteria']] = combined_df['eligibility_clean'].apply(split_eligibility)

    combined_df = combined_df[[id_col, summary_col, desc_col, 'inclusion_criteria', 'exclusion_criteria', 'intervention_name_clean', 'outcomes_name']]

    #rename columns so that they are returned same irrespective of source

    standardized_columns = {
        'identificationModule.nctId': 'nctId',
        'descriptionModule.briefSummary': 'briefSummary',
        'descriptionModule.detailedDescription': 'detailedDescription',
        'inclusion_criteria': 'inclusion_criteria',
        'exclusion_criteria': 'exclusion_criteria',
        'intervention_name_clean': 'intervention_name_clean',
        'outcomes_name': 'outcomes_name'
    }

    combined_df = combined_df.rename(columns=standardized_columns)
    desired_order = [
        'nctId',
        'briefSummary',
        'detailedDescription',
        'inclusion_criteria',
        'exclusion_criteria',
        'intervention_name_clean',
        'outcomes_name'
    ]

    combined_df = combined_df[desired_order]
    return combined_df


def extract_from_clinicaltrial(df):
    protocol_df = pd.json_normalize(df['protocolSection'])
    protocol_df = flatten_and_extract(protocol_df, 'interventions', 'primaryOutcomes')

    if protocol_df['nctId'].dropna().nunique() == 1:
        protocol_df['nctId'] = protocol_df['nctId'].fillna(protocol_df['nctId'].dropna().iloc[0])

    required = ['nctId', 'briefSummary', 'detailedDescription', 'eligibilityCriteria']
    protocol_df = ensure_columns(protocol_df, required)

    return aggregate_and_clean(protocol_df, 'nctId', 'briefSummary', 'detailedDescription', 'eligibilityCriteria')

def extract_from_clinicaltrials(df):
    protocol_df = pd.json_normalize(df['protocolSection'])
    protocol_df = flatten_and_extract(protocol_df, 'armsInterventionsModule.interventions', 'outcomesModule.primaryOutcomes')

    required = ['identificationModule.nctId', 'descriptionModule.briefSummary', 'descriptionModule.detailedDescription', 'eligibilityModule.eligibilityCriteria']
    protocol_df = ensure_columns(protocol_df, required)

    return aggregate_and_clean(
        protocol_df,
        'identificationModule.nctId',
        'descriptionModule.briefSummary',
        'descriptionModule.detailedDescription',
        'eligibilityModule.eligibilityCriteria'
    )

if __name__ == "__main__":
    # Get the PROJECT ROOT (biomed-extractor/)
    PROJECT_ROOT = 'c:\\Users\\USER\\Documents\\github\\biomed_extractor'

    # Data directory at top level
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

    df_csv = load_trials_csv(filepath = DATA_DIR, filename ='example_trials.csv')
    print(df_csv.head())

    df_json = load_trials_json(filepath = DATA_DIR, filename ='example_trials.json')
    #mydf = extract_elegibility_trtm_from_clinicaltrials(df_json)
    mydf = extract_from_clinicaltrials(df_json)
    print(mydf.head())
    # write processed json file to csv
    output_csv_path = os.path.join(DATA_DIR, 'example_trials_processed.csv')
    mydf.to_csv(output_csv_path, index=False)

    #load single file
    # Get the PROJECT ROOT (biomed-extractor/)
    PROJECT_ROOT = 'c:\\Users\\USER\\Documents\\github\\biomed_extractor'

    # Data directory at top level
    DATA_DIR_SINGLE_CSV = os.path.join(PROJECT_ROOT, 'data\\annotated\\ctg-studies_for_gold_individual_csv')
    df_csv = load_trials_csv(filepath = DATA_DIR_SINGLE_CSV, filename ='NCT00667810.csv')
    print(df_csv.head())

    # Data directory at top level
    DATA_DIR_SINGLE = os.path.join(PROJECT_ROOT, 'data\\annotated\\ctg-studies_for_gold_individual')
    #data\annotated\ctg-studies_for_gold_individual\NCT00667810.json

    df_json_single = load_trials_json(filepath = DATA_DIR_SINGLE, filename='NCT00667810.json')
    mydf_single = extract_from_clinicaltrial(df_json_single)
    print(mydf_single.head())
 