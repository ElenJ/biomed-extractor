# implement code to load and do inference with one model
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # suppresses Huggingface warning of storing data rather than symlinking it
#from transformers import AutoTokenizer, AutoModelForPreTraining, pipeline

import sys
# Insert the project root (biomed_extractor) into sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# load functions for import of clinicaltrials.gov data written previously
from app.data.loader import load_trials_json, extract_from_clinicaltrials
from utils import * # custom functions required for NER and summarization
from transformers import pipeline
import pandas as pd

# Example for NER
def load_ner_pipeline(model_name="kamalkraj/BioELECTRA-PICO"):
    ner = pipeline("token-classification", model=model_name, aggregation_strategy="simple")
    return ner


if __name__ == "__main__":
    # Get the PROJECT ROOT (biomed-extractor/)
    PROJECT_ROOT = 'c:\\Users\\elena.jolkver\\Documents\\github\\biomed_extractor'
    # Data directory at top level
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    df_json = load_trials_json(filepath = DATA_DIR, filename ='example_trials.json')
    mydf = extract_from_clinicaltrials(df_json)
    ner_pipeline = load_ner_pipeline()
    ner_res = process_trials_for_PICO(mydf, ner_pipeline)
    print("Your results are in!")
    print(ner_res.head())
    ner_res.to_csv(os.path.join(DATA_DIR, 'ner_results.csv'))
