# **Data Sources and APIs Used**

**Project:** Biomedical LLM Information Extraction Tool  
**Version:** 0.1  
**Author:** Elena Jolkver
**Date:** 25.07.2025



## **1. Primary Data Source**

### 1.1 ClinicalTrials.gov

- **Overview:**  
  ClinicalTrials.gov is a globally recognized, publicly accessible registry of clinical studies conducted around the world. It contains protocols, results summaries, and structured metadata for studies involving human participants.

- **URL:**  
  [https://clinicaltrials.gov/](https://clinicaltrials.gov/)

- **Data Formats Supported:**  
  - XML (recommended for structured ingestion)
  - Plain text summaries
  - [CSV/TSV exports](https://clinicaltrials.gov/ct2/resources/download)

- **Data Content:**  
  - Study identifier, title, sponsor, condition, interventions
  - Study design, eligibility criteria, locations
  - Primary and secondary outcomes
  - Summary results (when available)
  - Extensive metadata (e.g., enrollments, study phase, status)

- **Update Frequency:**  
  - ClinicalTrials.gov is continuously updated; data files reflect the current state of all registered studies at the time of download.

- **Access Method in Tool:**  
  - Users manually download XML files (single or bulk download from ClinicalTrials.gov) and upload them via the tool's interface.
  - (Future roadmap: Incorporate automated data fetching from ClinicalTrials.gov APIs.)

### 1.2 Training data

- **Overview:**  
  BIDS-Xu-Lab provide training files for PICO task. The associated publication "Towards precise PICO extraction from abstracts of randomized controlled trials using a section-specific learning approach" (Bioinformatics 2023 Sep 5;39(9)) is [here](10.1093/bioinformatics/btad542)

- **URL:**  
  [https://github.com/BIDS-Xu-Lab/section_specific_annotation_of_PICO/](https://github.com/BIDS-Xu-Lab/section_specific_annotation_of_PICO/)


- **Data Content:**  

This resource contains annotated abstracts from randomized controlled trials for PICO extraction tasks, including disease-specific corpora for Alzheimer's Disease and COVID-19, as well as a re-annotated subset from the EBM-NLP corpus.

The repo is also accessible as python PICO package. 


- **Access Method in Tool:**  

  - Used for model training only


## **2. Data Licensing and Compliance**

- Data from ClinicalTrials.gov is in the public domain, freely available for use, research, and redistribution without restriction.
- Users should ensure compliance with local privacy or ethical considerations when combining ClinicalTrials.gov data with proprietary datasets.
- Data from BIDS-Xu-Lab contains no info on licensing, but it is open source and published.



## **3. Other Data Sources (Planned/Future)**

- **PubMed/PMC:**  
  For literature abstracts and published study summaries.
- **Institutional Patient Data (if integrated):**  
  All integrations with proprietary or sensitive medical records will require appropriate ethical approvals and compliance checks.


## **4. APIs and External Services**

- **Current Version:**  
  - No external APIs are called by default. All data processing is performed locally on uploaded files to maximize privacy and reduce security risks. User can select huggingface models, which are fetched from the huggingface repo.

- **Planned Enhancements:**  
  - Option to fetch and update study data directly from ClinicalTrials.gov via their REST API or FTP bulk download endpoints.
  - Updates to permit secure, institution-limited use of cloud-based LLM APIs, fully controlled by user opt-in.



## **5. Data Preprocessing and Validation**

- Uploaded XML or text files are parsed, validated (schema and key fields checked), and sanitized before running extractions.
- Invalid or unsupported files are flagged with clear error messages to guide users.



## **6. References**

- [ClinicalTrials.gov: About the Data](https://clinicaltrials.gov/ct2/about-site/background)
- [ClinicalTrials.gov API Documentation](https://clinicaltrials.gov/data-api/api)
- [ClinicalTrials.gov Bulk Download](https://clinicaltrials.gov/ct2/resources/download)

