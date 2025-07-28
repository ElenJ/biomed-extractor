# **Data Sources and APIs Used**

**Project:** Biomedical LLM Information Extraction Tool  
**Version:** 0.1  
**Author:** Elena Jolkver
**Date:** 25.07.2025



## **1. Primary Data Source**

### ClinicalTrials.gov

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



## **2. Data Licensing and Compliance**

- Data from ClinicalTrials.gov is in the public domain, freely available for use, research, and redistribution without restriction.
- Users should ensure compliance with local privacy or ethical considerations when combining ClinicalTrials.gov data with proprietary datasets.



## **3. Other Data Sources (Planned/Future)**

- **PubMed/PMC:**  
  For literature abstracts and published study summaries.
- **Institutional Patient Data (if integrated):**  
  All integrations with proprietary or sensitive medical records will require appropriate ethical approvals and compliance checks.


## **4. APIs and External Services**

- **Current Version:**  
  - No external APIs are called by default. All data processing is performed locally on uploaded files to maximize privacy and reduce security risks.

- **Planned Enhancements:**  
  - Option to fetch and update study data directly from ClinicalTrials.gov via their REST API or FTP bulk download endpoints.
  - Potential integration with Huggingface Model Hub (for on-demand model fetching).
  - Updates to permit secure, institution-limited use of cloud-based LLM APIs, fully controlled by user opt-in.



## **5. Data Preprocessing and Validation**

- Uploaded XML or text files are parsed, validated (schema and key fields checked), and sanitized before running extractions.
- Invalid or unsupported files are flagged with clear error messages to guide users.



## **6. References**

- [ClinicalTrials.gov: About the Data](https://clinicaltrials.gov/ct2/about-site/background)
- [ClinicalTrials.gov API Documentation](https://clinicaltrials.gov/data-api/api)
- [ClinicalTrials.gov Bulk Download](https://clinicaltrials.gov/ct2/resources/download)

