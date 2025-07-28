# **Architecture Overview**

**Project:** Biomedical LLM Information Extraction Tool  
**Version:** 0.1  
**Author:** Elena Jolkver  
**Date:** 25.07.2025      
This document provides a high-level overview of the architecture for the Biomedical LLM Information Extraction Tool, detailing its components, interactions, and design principles. 



## **1. High-Level Architecture Diagram**

![Architecture Diagram](architecture_diagram.png)

This diagram illustrates the main components of the system, including the user interface, NLP processing pipeline, data handling, and deployment layer. Each component is designed to work together seamlessly to provide a robust information extraction tool for biomedical documents.

## **2. Component Overview**

### 2.1 User Interface (Streamlit Frontend)
- Provides a browser-based web UI for file upload, parameter selection, and results display.
- Handles user interactions and forwards data to the backend pipeline.

### 2.2 NLP Processing Pipeline
- Modular Python backend containing:
    - Data loading/cleaning logic for ClinicalTrials.gov files
    - Information extraction modules (PICO extractor, etc.)
    - LLM-based summarization module
    - Utilities for error handling and logging

### 2.3 Data Handling
- Accepts user-uploaded files through the Streamlit interface.
- Backend supports loading from local files; prepared for future integration with ClinicalTrials.gov API.

### 2.4 Model Management
- Pretrained models loaded at startup or on-demand (e.g., BioBERT, T5, etc.)
- Option for swapping models via configuration/UI in future versions.

### 2.5 Results Presentation
- Extracted results and summaries are rendered in real-time within the UI.
- Option to download results as CSV/Excel/JSON.

### 2.6 Deployment Layer
- All application components are packaged in a single Docker container for portability.
- Docker ensures consistent environment across local, on-prem, and cloud deployments.


## **3. Third-Party Dependencies**

- **NLP/ML Libraries:**  
  - Huggingface Transformers, Scikit-learn, pandas
- **Web/App Framework:**  
  - Streamlit
- **Deployment:**  
  - Docker (requirements.txt, Dockerfile)
- **Testing:**  
  - pytest


## **4. Extensibility Considerations**

- Modular code structure – new extractors, models, or output types can be added with minimal refactoring.
- API endpoints (for non-interactive integration) can be exposed by refactoring backend logic into a RESTful service (e.g., with FastAPI) in future versions.
- Designed for batch and single-document processing.



## **5. Security & Privacy**

- Data stays local to the deployment environment—no uploads to external infrastructure unless configured otherwise.
- Ready for containerized installation in high-compliance environments (e.g., healthcare, pharma).


## **6. Future Roadmap (Optional)**

- Multi-user support and authentication.
- Integration with external databases or literature APIs.
- Real-time Question Answering and feedback correction loop.



## **7. Change Log Reference**

See [changelog.md](changelog.md) for evolution and major architecture changes.

