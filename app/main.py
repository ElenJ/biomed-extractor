import streamlit as st
# load functions for import of clinicaltrials.gov data written previously
from data.loader import extract_from_clinicaltrials, extract_from_clinicaltrials_csv
import pandas as pd
from nlp.pipelines import load_ner_pipeline_huggingface, load_ner_trained_pipeline
from nlp.utils import * # import all utils
import altair as alt



# define functions
@st.cache_data
def process_extracted_data(df):
    # first, split strings into lists
    df = df.assign(
    intervention_extracted=df.intervention_extracted.str.split("; "),
        comparator_extracted=df.comparator_extracted.str.split("; "),
        outcome_extracted=df.outcome_extracted.str.split(r",\s*|;\s*")
    )
    # then explode lists into rows\n",
    for col in ['intervention_extracted', 'comparator_extracted', 'outcome_extracted']:
        df = df.explode(col)
    return df

@st.cache_data
def run_extraction(mydf_manual_annotation, model_selected):
    if model_selected == "bio-mobilebert":
        ner_pipeline = load_ner_trained_pipeline("app/model/nlpie_bio-mobilebert_PICO")
        ner_res_model = process_trials_for_retrained_PICO(mydf_manual_annotation, ner_pipeline)
    elif model_selected == "BioELECTRA":
        ner_pipeline = load_ner_pipeline_huggingface("kamalkraj/BioELECTRA-PICO")
        ner_res_model = process_trials_for_PICO(mydf_manual_annotation, ner_pipeline)
    else:
        ner_res_model = pd.DataFrame()
    clinicaltrials_pico = ner_res_model[["nctId", "summary_extracted", "intervention_extracted", "comparator_extracted", "outcome_extracted", "population_extracted"]]
    return clinicaltrials_pico

def plot_top_entities(
    df,
    entity_col,
    label,
    slider_key,
    max_n=10,
    default_n=5,
    chart_width=600,
    chart_height=400,
):
    """
    Plots a bar chart of the top N entities in a column, using an Altair chart in Streamlit.
    Returns nothing (writes result to Streamlit directly).
    """
    # Drop duplicates for nctId+entity, group, and count
    agg_df = (
        df.drop_duplicates(subset=['nctId', entity_col])
        .groupby(entity_col)
        .agg(count=(entity_col, 'size'))
        .sort_values(by='count', ascending=False)
    )
    # Remove empty
    agg_df = agg_df[agg_df.index != '']

    top_n = st.slider(f"Select top n {label.lower()}s", 0, max_n, default_n, step=1, key=slider_key)
    top_df = agg_df.head(top_n)

    chart = (
        alt.Chart(top_df.reset_index())
        .mark_bar()
        .encode(
            x=alt.X(f"{entity_col}", sort='-y', title=label),
            y=alt.Y('count', title='Number of Trials'),
            tooltip=[entity_col, 'count']
        )
        .properties(
            title=f"Top {top_n} {label.capitalize()}s in Clinical Trials",
            width=chart_width,
            height=chart_height,
        )
        .configure_axis(labelAngle=-90)
    )
    st.altair_chart(chart, use_container_width=True)

###########################
# The actual app
st.title("Biomed Extractor")
st.header("Extract biomedical entities from clinicaltrials.gov", divider="rainbow")

with st.sidebar:
    st.header("Control elements", divider="rainbow")
    st.markdown("Select files and control elements here")
    uploaded_file = st.file_uploader("Choose a csv or json file exported from clinicaltrials.gov")
    model_selected = st.selectbox(
    "Which model to use?",
    ("bio-mobilebert", "BioELECTRA"),
    index=None,
    placeholder="Select PICO model...",
    key="model_select"
    )
    st.write("You selected:", model_selected)

    st.header("Help and self-help", divider="rainbow")
    st.markdown("[User Documentation](https://github.com/ElenJ/biomed-extractor/blob/main/docs/user_manual.md)") 
    st.markdown("[Troubleshooting Guide](https://github.com/ElenJ/biomed-extractor/blob/main/docs/troubleshooting.md)")
    st.markdown("[Contact Support](https://github.com/ElenJ/biomed-extractor/issues)")

    #with open("docs/user_manual.pdf", "rb") as f:
     #   st.download_button("Download manual", f, file_name="BIOMED_EXTRACTOR_user_manual.pdf")





st.markdown("## File upload")
# Only process if both file and model are selected
if uploaded_file is not None and model_selected is not None:
    st.write("Filename: ", uploaded_file.name)
    file_extension = uploaded_file.name.split('.')[-1]
    if file_extension == 'json':
        df_json = pd.read_json(uploaded_file)
        mydf_manual_annotation = extract_from_clinicaltrials(df_json)
    elif file_extension == 'csv':
        df_csv = pd.read_csv(uploaded_file)
        mydf_manual_annotation = extract_from_clinicaltrials_csv(df_csv)
    else:
        st.error("Unsupported file type. Please upload a JSON or CSV file.")
        mydf_manual_annotation = pd.DataFrame()

    # Extraction trigger
    if st.button("Run Extraction"):
        with st.spinner("Extracting entities..."):
            clinicaltrials_pico = run_extraction(mydf_manual_annotation, model_selected)
            st.session_state['clinicaltrials_pico'] = clinicaltrials_pico

    # Only show results if extraction has been run
    if 'clinicaltrials_pico' in st.session_state:
        clinicaltrials_pico = st.session_state['clinicaltrials_pico']
        st.header("Results", divider="rainbow")
        st.write("Processed trials with extracted PICO elements:")
        processed_table = st.data_editor(clinicaltrials_pico, hide_index=True)

        st.markdown("## Overview")
        clinicaltrials_processed = process_extracted_data(clinicaltrials_pico)
        col1, col2, col3 = st.columns(3)
        with col1:
            plot_top_entities(clinicaltrials_processed, 'intervention_extracted', 'Intervention', slider_key='interv')
        with col2:
            plot_top_entities(clinicaltrials_processed, 'comparator_extracted', 'Comparator', slider_key='comp')
        with col3:
            plot_top_entities(clinicaltrials_processed, 'outcome_extracted', 'Outcome', slider_key='outc')

        st.download_button(
            label="Download Results as tab-separated CSV",
            data=clinicaltrials_pico.to_csv(index=False, sep='\t').encode('utf-8'),
            file_name="extracted_trials.csv",
            mime="text/csv"
        )


else:
    st.info("Please upload a file and select a model to see analysis!") 
