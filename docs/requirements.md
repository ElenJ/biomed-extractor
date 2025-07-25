# **Biomedical LLM Information Extraction Tool Requirements Specification**


**Project:** Biomedical LLM Information Extraction Tool  
**Version:** 0.1  
**Author:** Elena Jolkver 
**Date:** 25.07.2025


## **1. Introduction**

This document details the functional and non-functional requirements for the Biomedical LLM Information Extraction Tool, defining the expected features, behavior, and constraints of the application.


## **2. Functional Requirements**

### 2.1 User Interface

- **FR1. File Upload:**  
  Users must be able to upload one or more ClinicalTrials.gov documents (XML or text format) through the web UI.
- **FR2. Extraction Trigger:**  
  Users must be able to trigger extraction of key information (e.g., Population, Intervention, Comparator, Outcome – PICO) from uploaded files.
- **FR3. Results Display:**  
  The tool must display extracted elements and summaries in a structured, easy-to-read table or report.
- **FR4. Download Results:**  
  Users must be able to download results (e.g., as CSV, Excel, or JSON).
- **FR5. Error Messaging:**  
  The application must report errors in user-friendly language if data is missing, incorrectly formatted, or an extraction step fails.

### 2.2 Information Extraction

- **FR6. PICO Element Extraction:**  
  The tool must extract Population, Intervention, Comparator, and Outcomes from clinical trial documents using an LLM-based or hybrid pipeline.
- **FR7. Summarization:**  
  The tool must generate a concise, LLM-generated summary of each processed document.
- **FR8. Multi-document Batch Processing:**  
  The tool should support processing multiple documents in one session and summarizing results in aggregate.
- **FR9. Configuration:**  
  Users should be able to select different extraction models (if available) via UI options.

### 2.3 Extensibility & Integration

- **FR10. API-ready Backend:**  
  The backend must be designed modularly to allow for API exposure in the future.
- **FR11. Dockerized Deployment:**  
  The application must be runnable in a Docker container, supporting local/private deployment.


## **3. Non-Functional Requirements**

- **NFR1. Usability:**  
  The web interface must require no coding skills and should be usable by domain experts (not just developers).
- **NFR2. Performance:**  
  For batch uploads of up to 10 documents, extractions and summaries should return in under 5 minutes.
- **NFR3. Security & Privacy:**  
  Uploaded data must not be sent to external servers (unless using public LLM APIs is explicitly enabled by the user).
- **NFR4. Portability:**  
  The application must run on Windows, Mac, and Linux (either via Docker or natively with Python 3.11+).
- **NFR5. Documentation:**  
  Clear setup, usage, and deployment documentation must be included in the repository.
- **NFR6. Testing:**  
  A suite of automated tests must be provided for core functions.


## **4. Out of Scope for V0.1**

- Integration with production external databases or EMRs.
- Processing and extracting from non-English documents.
- Real-time, large-scale batch processing (>100 documents at once).
- Advanced authentication or user management features.


## **5. Acceptance Criteria**

- All functional requirements successfully demonstrated through manual testing and/or automated tests.
- Dockerized version builds and runs locally with sample files.
- End-user test with at least 5 trial documents results in interpretable extractions/outputs.
- Clear documentation and error handling.


## **6. Future Enhancements (Optional)**

- Add interactive feedback (user can correct or validate extracted elements in-app).
- Integrate Question Answering (QA) features for user-driven queries on document sets.
- Expand source formats (e.g., support for PubMed).
- Deep integration with organizational user authentication and logging.

