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
│   │   └── dmis-lab_biobert-v1.1/ # biobert finetuned for PICO
│   └── config.py            # Central (editable) config
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
│   ├── goals_and_use_cases.docx # Document outlining project goals, use cases, and target audience
│   ├── goals_and_use_cases.pdf # PDF version of the goals and use cases document
│   ├── requirements.md        # Detailed requirements and specifications
│   ├── architecture.md       # High-level architecture overview
│   ├── design.md             # Design decisions and rationale  
│   ├── user_interface.md      # User interface design and user experience considerations
│   ├── data_sources.md       # Information on data sources and APIs used
│   ├── model_selection.md    # Model selection process and rationale
│   ├── evaluation.md         # Evaluation metrics and methods
│   ├── deployment.md         # Deployment instructions and considerations
│   ├── testing.md            # Testing strategy and test cases
│   ├── user_manual.md        # User manual for the tool
│   ├── troubleshooting.md    # Common issues and troubleshooting steps
│   ├── changelog.md          # Change log for version history
│   ├── architecture_diagram.png # Visual representation of the system architecture
│   ├── design_diagram.png    # Visual representation of the design
│   ├── user_interface_mockup.png # Mockup of the user interface
│   ├── data_flow_diagram.png # Data flow diagram showing how data moves through
│   ├── model_architecture.png # Diagram of the model architecture
│   ├── evaluation_metrics.png # Visual representation of evaluation metrics
│   ├── deployment_diagram.png # Diagram showing deployment architecture
│   ├── testing_strategy.png   # Visual representation of the testing strategy
│   ├── pics/ # Screenshots for md files
│   │   ├── architecture_diagram.png  # Overvieew on app architecture
│   │   ├── upload_screen.png  # Screenshot of the upload screen
│   │   └── results_screen1.png # Screenshot of the results screen (upper part)
│   ├── troubleshooting_examples/ # Examples of common issues and solutions
│   │   ├── issue_1.md         # Example of a common issue and its solution
│   │   ├── issue_2.md         # Another common issue and its solution
│   │   └── issue_3.md         # Yet another common issue and its solution
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

## Project Plan

This plan assumes you’ll spend about 10–15 hours a week on the project and can be adjusted based on your pace and resources. The goal is to have a working prototype by the end of 8–10 weeks, with clear milestones and check-ins to keep you on track.


## **LLM-Powered Clinical Study Information Extraction Tool**
**Goal:** Develop a prototype tool that allows users to upload clinical trial abstracts or biomedical documents and automatically extracts key information (e.g., PICO elements) and generates structured summaries. The tool will have a user-friendly interface (Streamlit/Gradio or RShiny).



### **Phase 1: Scoping and Foundation (1 week)**
- **Define goals, audiences, and main use cases**
- **Decide on tech stack** (Python with Streamlit/Gradio OR R with Shiny)
- **Select initial dataset(s)** (e.g., PubMed abstracts, public COVID19 trial data, ClinicalTrials.gov exports)
#### **Check-in 1:** Clearly written project scope, target users, and outline of technical approach.
- **Task achieved**: goals and use cases document as well as initial architecture and design documents written, initial dataset identified (ClinicalTrials.gov), tech stack chosen (Python/Streamlit).
TODO: check documents once tool programming is done (some were created, but the text is rather dummy-like; change xml to json everywhere)


### **Phase 2: Data Collection & Preparation (1 week)**
- **Acquire sample documents** (10–100 abstracts/trials for rapid iteration; can scale up later)
- **Create or locate gold-standard annotations** (small set where PICO elements are labeled; can start by annotating a small batch yourself)
- **Basic text preprocessing** (cleaning, segmenting, field extraction)
#### **Check-in 2:** Dataset prepared, previewed in Jupyter/R Notebook; sample annotation table.


### **Phase 3: Model Selection & Prototyping (2–3 weeks)**
- **Experiment with ready-made transformer models**  
  (Start with huggingface pipelines for NER/extraction using BioBERT/ClinicalBERT, or generic BERT if needed)
- **Prototype information extraction**  
  (initial extractors for Population/Intervention/Comparator/Outcome via prompt or finetuned NER/classifier)
- **Prototype text summarization** (using models like T5, BioGPT, or BART)
#### **Check-in 3:** Jupyter notebook (or R notebook) with initial extraction and summarization results on your dataset.


### **Phase 4: Model Refinement & Evaluation (2 weeks)**
- **Improve extraction by fine-tuning (if resources permit), prompt engineering, or post-processing rules**
- **Build evaluation scripts** (compare extraction/summaries to ground truth; report precision/recall or qualitative evaluation)
- **Draft user-facing explanations about model confidence and limitations**
#### **Check-in 4:** Evaluation notebook/code; basic writeup (README or md doc) of performance and sample outputs with discussion.



### **Phase 5: App & Interface Development (2 weeks)**
- **Design UI/UX mockup:** Flow from document upload to extraction & summary presentation.
- **Build MVP UI** (Streamlit/Gradio/RShiny prototype where user uploads docs and sees results)
- **Integrate model into interface** so it processes real-time uploads
- **Handle errors, edge cases, and input variations**
#### **Check-in 5:** Working prototype app; can be demonstrated live to a user.



### **Phase 6: Packaging, Documentation & Sharing (1 week)**
- **Document installation and usage instructions (README.md)**
- **Include test/sample documents**
- **Add screenshots/gifs and a short video demo if possible**
- **Write a blog post or LinkedIn article explaining the tool, its use case, and technical highlights**
#### **Final Check-in:** Fully documented public GitHub repo + demo material ready for sharing with potential clients or as an educational case study.



### **Optional Stretch Goals:**
- Deploy with Docker for easy private/in-house setup.
- Add interactive corrections: allow user to edit outputs and track feedback.
- Incorporate QA functionality (“Ask a question about the documents”).
- Track usage, feedback, or error logs.



### **Self-Check Questions for Each Phase**
- **Is the main use case still clear and valuable?**
- **Can another user (e.g., a researcher) successfully run/install the prototype?**
- **Does the interface clearly present results and limitations?**
- **Are there at least 1–2 evaluated metrics, and are sample outputs interpretable?**



**Remember:**  
Update your README every 1–2 weeks with progress and learning reflections—this not only helps keep you on track, it also generates content for your eventual project blog post or book chapter.

