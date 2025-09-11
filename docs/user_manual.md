# **User Manual**

**Biomedical LLM Information Extraction Tool**  
**Version:** 0.1  
**Author:** Elena Jolkver
**Date:** 11.09.2025



## **1. Introduction**

Welcome! This tool provides an easy-to-use interface for extracting key information—such as Population, Intervention, Comparator, and Outcomes (PICO)—and generating summaries from ClinicalTrials.gov documents, using advanced AI models tailored for biomedical data.

The tool is designed for clinical researchers, medical staff, and data analysts who want to quickly review or process multiple clinical trial files without programming.



## **2. System Requirements**

- **Hardware:** Windows, Mac, or Linux; minimum 4 CPU cores and 8GB RAM recommended.
- **Software:** Modern web browser (Chrome, Firefox, Edge, Safari).  
  No additional software is needed if provided as a Docker or web service.


## **3. Getting Started**

**If running locally via Docker:**
1. Install [Docker](https://www.docker.com/get-started).
2. In your terminal, go to the project directory and run:

    ```bash
    docker build -t biomed-extractor .
    docker run -p 8501:8501 biomed-extractor
    ```

3. Open your browser and go to:  
   [http://localhost:8501](http://localhost:8501)

**If accessed via web portal:**  
Ask your system administrator or provider for the tool URL and open it in your browser.



## **4. Uploading Files**

1. On the main screen, find the **“Control elements”** section.
2. Click the **“Browse files”** button.
3. Select a ClinicalTrials.gov files (json or csv). The file might contain one or more trials. You can use biomed_extractor\data\example_trials.json as an example
    - Only supported formats will be accepted.
    - Drag and drop is also supported.
4. Uploaded file name will appear below the uploader.



## **5. Configuring Extraction (Optional)**

- You can choose extraction options (which AI model to use) in the sidebar.


## **6. Running Extraction**

1. After uploading files, click the **"Run Extraction"** button.
2. The tool will process the documents. Progress and status will be shown.
3. Wait for completion—processing time depends on model selected and document size/number.



## **7. Viewing Results**

- Extracted information and AI-generated summaries will be displayed in a table on the results screen.
- Top 5 Interventions, comparators and outcomes are displayed in bar charts.



## **8. Downloading Results**

- Use the **"Download"** button to obtain results as a tab-separated csv file.
- The download will include all extraction results and summaries for your uploaded files.



## **9. Error Handling and Troubleshooting**

- If you see an error message:
  - Check file format and ensure the documents conform to ClinicalTrials.gov standards.
  - Large documents or batches may take longer; verify your system meets minimum specifications.
  - For persistent issues, see the [Troubleshooting Guide](troubleshooting.md) or contact your administrator.



## **10. User Privacy and Security**

- All processing is local to your system unless otherwise specified.
- Your uploaded data is **not sent to external servers**.
- For private/publication-sensitive data, deploy the tool only on secure, authorized machines.


## **11. Frequently Asked Questions (FAQ)**

**Q: What document formats are supported?**  
A: ClinicalTrials.gov json and plain text csv files. For other formats, convert to supported types before uploading.


**Q: How do I select a different AI model?**  
A: Model selection options are in the dropdown menu in the sidebar.

**Q: Where can I get help?**  
A: See the help links in the sidebar.



## **12. Support and Feedback**

- Submit an issue via the project’s [GitHub repository](https://github.com/ElenJ/biomed-extractor) for questions, suggestions, or reporting a bug.
- User feedback is encouraged to improve future versions of the tool!


*For further details on the tool’s purpose, design, or technical background, see the project’s documentation in the `docs/` folder.*