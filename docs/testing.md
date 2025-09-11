# **Testing Strategy and Test Cases**

**Project:** Biomedical LLM Information Extraction Tool**  
**Version:** 0.1  
**Author:** Elena Jolkver
**Date:** 11.09.2025


## **1. Introduction**

This document outlines the testing strategy, types of tests, test automation setup, and example test cases for the Biomedical LLM Information Extraction Tool. Rigorous testing ensures correctness, reliability, and maintainability of both the application’s business logic and user interface.



## **2. Testing Strategy**

### **2.1. Test Types**

- **Unit Tests:**  
  Test individual functions and modules (e.g., data loaders, text cleaners, extraction logic).
- **Integration Tests:**  
  Test interactions between modules—such as running data through the full extraction pipeline and checking combined outputs.
- **End-to-End (E2E) Tests:**  
  Simulate typical user workflows: file upload, extraction, reviewing results, and downloads through the Streamlit interface.
- **Regression Tests:**  
  Ensure previously fixed bugs do not reappear.
- **Performance & Load Tests:**  
  Assess how the tool performs with larger document batches or complex documents.

### **2.2. Testing Tools**

- **Framework:** `pytest`
- **Mocking:** not used
- **Test Data:** inline data
- **Code Coverage:** `pytest-cov` pytest --cov=app --cov-report=term-missing  
                                  pytest --cov=app --cov-report=xml
- **CI:** GitHub Actions runs all tests on each push and pull request, defined in ci.yaml



## **3. Test Automation**

- All unit and integration tests are located in the `/tests/` directory.
- Automated via GitHub Actions workflow (`.github/workflows/ci.yml`).
- Code coverage targets are monitored (aim for >80% where practical).


### **4. Test File Overview**

- `test_loader.py`: Tests data loading and parsing from ClinicalTrials.gov files.
- `test_ner_utils.py`: Tests text preprocessing and NER utility functions.
- `test_evaluate_model.py`: Tests model evaluation and metrics calculation.
- `test_training.py`: Tests training routines and model saving/loading.

## **4. Example Test Cases**

### **4.1. Unit Tests**

Representative test cases:

- **Test:** Data Loader parses valid ClinicalTrials.gov file  
  - **Given:** A well-formed example JSON or CSV file  
  - **When:** The loader function is called  
  - **Then:** The returned DataFrame contains all expected fields (e.g., NCT ID, title, population, interventions, outcomes) with correct values

- **Test:** Text Preprocessing removes HTML tags and normalizes whitespace  
  - **Given:** Text containing HTML markup and irregular spacing  
  - **When:** The preprocessing function is applied  
  - **Then:** The output is clean, plain text with no HTML tags and normalized spaces

- **Test:** Entity Deduplication removes duplicates and substrings  
  - **Given:** A list of intervention entities with duplicates and overlapping substrings  
  - **When:** The deduplication function is called  
  - **Then:** The output contains only unique, non-overlapping entities



### **4.2. Integration Tests**

Representative test cases:

- **Test:** Full Information Extraction Pipeline  
  - **Given:** A sample annotated ClinicalTrials.gov file  
  - **When:** The file is processed through the extraction pipeline  
  - **Then:** The extracted PICO elements (Population, Intervention, Comparator, Outcome) match the expected ground truth

- **Test:** Summarization Pipeline  
  - **Given:** A study description with multiple sentences  
  - **When:** The summarization function is run  
  - **Then:** The output summary is concise, relevant, and contains the main study points

- **Test:** Model Evaluation Metrics Calculation  
  - **Given:** Gold standard and predicted outputs for PICO elements  
  - **When:** The evaluation function is called  
  - **Then:** Precision, recall, and F1 scores are correctly calculated and reported




### **4.3. End-to-End (E2E) Tests**

Representative test cases, performed manually prior to release of new version:

- **Test:** User uploads file and downloads results via Streamlit  
  - **Given:** The application is running locally  
  - **When:** The user uploads a valid file, selects a model, triggers extraction, and downloads the results  
  - **Then:** The result file is created, contains correctly structured data, and no errors occur during the workflow

- **Test:** UI Error Handling  
  - **Given:** The user uploads an unsupported file type or leaves required fields empty  
  - **When:** The extraction is triggered  
  - **Then:** The application displays appropriate error messages and does not crash

### **4.4. Regression Tests**

Regression on fixed annotation bug (when identified)
A known edge-case file that previously triggered a bug (e.g., missing population field, malformed intervention data)  


### **4.5. Performance Tests**

**Test:** Batch Processing  
- **Given:** 10 varied clinical trial documents  
- **When:** Processed together  
- **Then:** Results are produced in <5 minutes, and all files are handled without crash



## **5. Manual Testing Checklist**

- Confirm UI error messages appear for unsupported files, empty uploads, or processing failures
- Test all download buttons and verify output format and content
- Review outputs for a variety of trial types (phases, sponsors, interventions, etc.)
- Verify Docker deployment does not affect core features or data paths



## **6. Quality Assurance and Maintenance**

- Expand test coverage with all new features and bug-fixes (Regression Tests)
- Code reviews and peer testing for all pull requests
- Regularly review automated test results and resolve any red (failing) tests before release
- Update `/tests/` with any new edge-case files encountered during user feedback

