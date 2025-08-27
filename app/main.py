import streamlit as st
# load functions for import of clinicaltrials.gov data written previously
from data.loader import extract_from_clinicaltrials, extract_from_clinicaltrials_csv
import pandas as pd
from nlp.pipelines import load_ner_pipeline
from nlp.utils import (
    compose_trial_text, chunk_text_by_chars, run_ner_on_long_text, clean_population_entities,
    merge_entities, extract_pico_from_merged_entities, normalize_intervention, is_substring_duplicate,
    deduplicate_intervention_entities, summarize_textRank, extract_comparator, remove_comparator_terms,
    clean_outcomes, process_trials_for_PICO
)

st.title("Biomed Extractor")
st.header("Extract biomedical entities and relations from clinicaltrials.gov")

with st.sidebar:
    st.header("Control elements")
    st.markdown("Select files and control elements here")
    uploaded_file = st.file_uploader("Choose a file")

col1, col2 = st.columns(2)

with col1:
    st.markdown("## File upload")


    if uploaded_file is not None:
        # find out file extension (json or csv)
        st.write("Filename: ", uploaded_file.name)
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension == 'json':
            # load data to extract info from
            df_json = pd.read_json(uploaded_file) 
            mydf_manual_annotation = extract_from_clinicaltrials(df_json)
        elif file_extension == 'csv':
            df_csv = pd.read_csv(uploaded_file)
            mydf_manual_annotation = extract_from_clinicaltrials_csv(df_csv)
        else:
            st.error("Unsupported file type. Please upload a JSON or CSV file.")
            mydf_manual_annotation = pd.DataFrame()

        # display dataframe
        # imported_table = st.data_editor(mydf_manual_annotation, hide_index=True)    
        # process file for PICO elements
        ner_pipeline = load_ner_pipeline()
        ner_res_model = process_trials_for_PICO(mydf_manual_annotation, ner_pipeline)
        # get extracted columns only
        #clinicaltrials_pico = ner_res_model[["nct_id", "extracted_population", "extracted_intervention", "extracted_comparator", "extracted_outcome"]]
        st.write(ner_res_model.columns)
        st.write("Processed trials with extracted PICO elements:")
        processed_table = st.data_editor(ner_res_model, hide_index=True)
with col2:
    st.markdown("## Overview")