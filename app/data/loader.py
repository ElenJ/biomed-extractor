import os
import pandas as pd
import re
import unicodedata

# Get the PROJECT ROOT (biomed-extractor/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data directory at top level
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def load_trials_csv(filename='example_trials.csv'):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} records from {filename}")
    return df

def load_trials_json(filename='example_trials.json'):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_json(path)
    print(f"Loaded {len(df)} records from {filename}")
    return df

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



def extract_elegibility_trtm_from_clinicaltrials(df):
    """    Extracts eligibility and treatment information from clinical trials DataFrame. """
    # Normalize from json to pandas df
    protocol_df = pd.json_normalize(df['protocolSection']) 
    # Apply flattening using .loc
    protocol_df.loc[:, 'interventions_clean'] = protocol_df['armsInterventionsModule.interventions'].apply(
        lambda x: flatten_list_of_dicts(x, ['type', 'name', 'description'])
    )

    protocol_df.loc[:, 'primaryOutcomes_clean'] = protocol_df['outcomesModule.primaryOutcomes'].apply(
        lambda x: flatten_list_of_dicts(x, ['measure', 'description', 'timeFrame'])
    )
    # Explode each list-based column
    protocol_df = protocol_df.explode('interventions_clean') \
                        .explode('primaryOutcomes_clean') \
                        .reset_index(drop=True)
    # extract relevant information from the cleaned columns
    protocol_df['intervention_name'] = protocol_df['interventions_clean'].str.extract(r'name:\s*(.*?)\s*;?\s*description:')
    protocol_df['outcomes_name'] = protocol_df['primaryOutcomes_clean'].str.extract(r'measure:\s*(.*?)\s*;?\s*description:')
    protocol_df['intervention_name_clean'] = protocol_df['intervention_name'].str.replace(r'\s*\d+\s*(mg|mcg|g|ml)', '', case=False, regex=True) 


    # Group by study ID and aggregate other columns
    combined_df = protocol_df.groupby('identificationModule.nctId').agg({
        'descriptionModule.briefSummary': lambda x: '; '.join(pd.unique(x.dropna())),
        'eligibilityModule.eligibilityCriteria': lambda x: '; '.join(pd.unique(x.dropna())),
        'intervention_name_clean': lambda x: '; '.join(pd.unique(x.dropna())),
        'outcomes_name': lambda x: '; '.join(pd.unique(x.dropna()))
    }).reset_index()  

    # Apply the cleaning function to the column
    combined_df['eligibility_clean'] = combined_df['eligibilityModule.eligibilityCriteria'].apply(clean_text)

    # Apply the function to split the column
    combined_df[['inclusion_criteria', 'exclusion_criteria']] = combined_df['eligibility_clean'].apply(split_eligibility)

        # Drop unnecessary columns cleaned DataFrame
    cleaned_df = combined_df[[
        'identificationModule.nctId',
        'descriptionModule.briefSummary',
        'inclusion_criteria', 
        'exclusion_criteria',
        'intervention_name_clean',
        'outcomes_name'
    ]].copy()

    return cleaned_df

if __name__ == "__main__":
    df_csv = load_trials_csv()
    print(df_csv.head())

    df_json = load_trials_json('example_trials.json')
    mydf = extract_elegibility_trtm_from_clinicaltrials(df_json)
    print(mydf.head())