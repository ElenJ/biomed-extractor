# **Design Decisions and Rationale**

**Project:** Biomedical LLM Information Extraction Tool  
**Version:** 0.1  
**Author:** Elena Jolkver
**Date:** 25.07.2025


## **1. Technology Stack**

### **1.1 Streamlit for Frontend/UI**
**Decision:**  
Use [Streamlit](https://streamlit.io/) to implement the web-based user interface.

**Rationale:**  
- Enables rapid development and iteration with minimal boilerplate.
- Target users (clinical researchers, bioinformaticians) often lack software engineering backgrounds; Streamlit’s UI is easy to navigate and requires no installation for the user.
- Facilitates real-time visualization and file uploads without needing web development expertise.

**Alternatives Considered:**  
- **FastAPI:** More suited for programmatic API or production backend, less for interactive UIs aimed at non-developers.
- **Dash, Flask:** More complex for our target quick prototyping needs.


### **1.2 Modular Python Backend**

**Decision:**  
Structure all application logic (data loading, NLP processing, extraction, summarization) in a modular `/app` directory, following best practices for extensible Python projects.

**Rationale:**  
- Modular code enables separation of concerns (UI, data ingest, NLP/model logic).
- Simplifies testing, maintenance, and feature extension (e.g., swapping models, adding new extraction tasks).
- Eases future integration of backend logic with APIs or other frontends.


### **1.3 Use of Pre-trained NLP Models (LLMs)**

**Decision:**  
Leverage Huggingface Transformers library for NLP tasks, starting with domain-adapted models (e.g., BioBERT/T5).

**Rationale:**  
- Pre-trained models significantly reduce time-to-solution while delivering strong performance on clinical/biomedical text.
- Huggingface ecosystem provides standard APIs, extensive documentation, and ongoing security/usability enhancements.
- Domain-specific models improve extraction and summarization fidelity compared to generic LLMs.

**Alternatives Considered:**  
- Rule-based methods (limited accuracy and scalability).
- Training custom models from scratch (unrealistic resource and data requirements at this stage).


### **1.4 Data Input and Processing**

**Decision:**  
Support upload of standard ClinicalTrials.gov documents (XML or text) via the UI, with batch processing support.

**Rationale:**  
- ClinicalTrials.gov is a widely used, structured, and openly accessible data source in clinical research.
- Batch upload removes manual bottlenecks and is essential for systematic review workflows.


### **1.5 Output and Reporting**

**Decision:**  
Display extracted elements and generated summaries in both tabular and human-readable form, with options to download results.

**Rationale:**  
- Target users must quickly interpret and reuse outputs in external tools (Excel, R, downstream analysis).
- Direct download enables easy integration into their workflow.


### **1.6 Deployment via Docker**

**Decision:**  
Package the application as a Docker container for easy cross-platform deployment (Windows, Mac, Linux) and local/private execution.

**Rationale:**  
- Many users (e.g., pharma, clinical sites) require assurance that sensitive data does not leave their local environment.
- Docker ensures consistent environment/reproducibility, reduces setup friction, and is a standard in both dev and clinical informatics.

**Alternatives Considered:**  
- Pure cloud-based deployment (potential privacy concerns and more infrastructure overhead).
- Native install scripts (environment/versioning challenges).


### **1.7 Testing and CI/CD**

**Decision:**  
Automated testing with `pytest` and Continuous Integration (CI) with GitHub Actions.

**Rationale:**  
- Early bug detection, repeatable quality checks, and easier onboarding for collaborators.
- GitHub Actions provides a zero-cost, scalable CI platform tightly coupled with the codebase.


## **2. Extensibility and Future-Proofing**

- All core pipelines (extraction, summarization) are loosely coupled—future models or tasks can be swapped or added with minimal refactoring.
- Backend logic is being built ready for API exposure, so can be connected to other frontends (or even mobile apps) in the future.
- Configuration files enable model selection, adjustable thresholds, etc., by power users.


## **3. Security and Privacy**

- By default, no data leaves the user’s local system when running via Docker.
- No 3rd-party APIs are called for model inference unless explicitly enabled/configured.


## **4. Out-of-Scope for Initial Version**

- Multi-user authentication
- Large dataset auto-ingestion or database links
- Real-time operation at massive scale
- Advanced document annotation/correction interfaces


## **5. Design Decisions Review**

This document will be periodically reviewed as new requirements or constraints arise; please see [changelog.md](changelog.md) for an audit of significant design changes.

