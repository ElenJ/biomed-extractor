# biomed-extractor
Biomedical Document Assistant: An LLM-Powered Information Extraction and Summarization Tool for Clinical Studies


Automatically extract PICO elements & structured summaries from ClinicalTrials.gov data.

## Features
- Upload clinical trial data
- Extract population/intervention/comparator/outcome
- Summarize trial
- Web-based interface (Streamlit)
- Ready for Docker deployment

## Quickstart

```bash
docker build -t biomed-extractor .
docker run -p 8501:8501 biomed-extractor
```

## Project structure

Files used for development of the nlp pipeline: xx.py

Files used for demo purposes: xx.ipynb

Use main.py in app/ to run the finalized app.

```
biomed-extractor/
├── .github/
│   └── workflows/           # CI/CD setup (e.g. GitHub Actions for testing)
├── app/                     # All application (tool) code
│   ├── __init__.py
│   ├── main.py              # Entrypoint for Streamlit/FastAPI/etc.
│   ├── nlp/                 # NLP/model logic: data processing, inference
│   │   ├── __init__.py
│   │   ├── pipelines.py    # Huggingface pipelines for NER/summarization
│   │   ├── train_model.py # Script to fine-tune a pretrained transformer model on PICO NER task
│   │   ├── training_PICO_model_on_colab.ipynb # Jupyter notebook used to fine-tune a pretrained transformer model on PICO NER task on Google COLAB (for GPU usage)
│   │   ├── utils.py         # Utility functions for data processing and model inference
│   │   ├── evaluate_model.py # Evaluation logic for model outputs
│   │   ├── compare_model_performance.ipynb # Comparison of self-finetuned models and model selection for app
│   │   ├── demo_inference.ipynb    # Jupyter notebook showing model inference and evaluation
│   │   ├── training_PICO_model_development.ipynb # Jupyter notebook used to develop model training
│   │   └── development_inference.ipynb # notebook used to develop bulk of the functions and the pipeline
│   ├── data/                # Data loading and ClinicalTrials.gov API code
│   │   ├── __init__.py
│   │   ├── get_pico_dataset.py # merging of foreign datasets for PICO training
│   │   ├── data_cleaning.ipynb # Jupyter notebook demonstrating data cleaning
│   │   └── loader.py # script to load clinicaltrials data
│   ├── model/                # Trained models
│   │   ├── nlpie_bio-mobilebert_PICO/ # mobilebert finetuned for PICO
│   │   ├── nlpie_compact_biobert_PICO/ # compact-biobert finetuned for PICO
│   └── └── dmis-lab_biobert-v1.1/ # biobert finetuned for PICO
├── tests/                   # Unit/integration tests
│   ├── __init__.py
│   ├── test_evaluate_model.py # tests for functions in nlp/evaluate_model
│   ├── test_training.py # tests for functions collected in nlp/train_model.py
│   ├── test_ner_utils.py # tests for functions collected in nlp/utils.py (for inference)
│   └── test_loader.py # tests for functions in data/loader.py
├── data/                    # Sample input data
│   ├── annotated/                 # NLP/model logic: data processing, inference
│   │   ├── gold_standard.json # human-extracted PICO elements from a subset of trials (ctg-studies_for_gold)
│   │   ├── ctg-studies_for_gold.json/.csv # studies used to generate gold_standard.json
│   │   └── ctg-studies_for_gold_individual/ # folder with gold-standard studies, stored individually
│   ├── example_trials.csv # Example ClinicalTrials.gov CSV data
│   ├── example_trials.json # Example ClinicalTrials.gov data
│   └── pico_dataset_for_training/ # PICO dataset used for training from https://github.com/BIDS-Xu-Lab/section_specific_annotation_of_PICO/
├── docs/
│   ├── architecture.md                  # High-level architecture overview
│   ├── changelog.md                     # Change log for version history
│   ├── data_sources.md                  # Information on data sources and APIs used
│   ├── deployment.md                    # Deployment instructions and considerations
│   ├── design.md                        # Design decisions and rationale  
│   ├── evaluation.md                    # Evaluation metrics and methods
│   ├── goals_and_use_cases.docx         # Document outlining project goals, use cases, and target audience
│   ├── goals_and_use_cases.pdf          # PDF version of the goals and use cases document
│   ├── model_selection.md               # Model selection process and rationale
│   ├── requirements.md                  # Detailed requirements and specifications
│   ├── testing.md                       # Testing strategy and test cases
│   ├── troubleshooting.md               # Common issues and troubleshooting steps
│   ├── user_interface.md                # User interface design and user experience considerations
│   ├── user_manual.md                   # User manual for the tool
│   ├── pics/ # Screenshots for md files
│   └── troubleshooting_examples/ # Examples of common issues and solutions
├── Dockerfile               # Docker container build instructions
├── requirements.txt         # Python dependencies
├── README.md                # Main documentation with setup, usage, and background
├── LICENSE
└── .gitignore

```
## Workflow for adaptation

1. Fine-tune your foundation model of choice with the PICO dataset (train_model.py)
2. Evaluate your model with evaluate_model.py
3. Adjust the app (main.py) to use your model of choice 

## Run app

```bash
streamlit run app/main.py
```

Demo app with file data/example_trials.json or data/example_trials.csv

Stop execution with Ctrl+C

## Technical background

Model selection and finetuning for PICO task as well as model performance is outlined in [model_selection.md](docs/model_selection.md). 

### PICO extraction

**Population**: Extracted by running a NER pipeline on inclusion criteria, when using models from Huggingface. Only entities containing demographic/diagnosis keywords or sufficiently long phrases are retained and de-duplicated. When running PICO-ectraction on own models (fine-tuned from base models), population extraction is performed on main trial text.

**Intervention/Outcome**: Extracted using NER on the main trial text (composed of briefSummary + detailedDescription) and cleaned by normalization, substring/fuzzy matching, and removal of generic or comparator terms.

**Comparator**: Extracted not by NER but by searching (using regular expressions or string matching) for a fixed list of comparator keywords (e.g., placebo, sham, usual care) within the intervention text.

**Summary**: Extracted by creating a brief extractive 2-sentence TextRank summary from the combined briefSummary and detailedDescription fields. 


