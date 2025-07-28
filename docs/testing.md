TODO: check document once some programming is done

# **Testing Strategy and Test Cases**

**Project:** Biomedical LLM Information Extraction Tool**  
**Version:** 0.1  
**Author:** [Your Name]  
**Date:** [Date]


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
- **Mocking:** `unittest.mock`, `pytest-mock`
- **Test Data:** Located under `/tests/test_data/`
- **Code Coverage:** `pytest-cov`
- **CI:** GitHub Actions runs all tests on each push and pull request



## **3. Test Automation**

- All unit and integration tests are located in the `/tests/` directory.
- Automated via GitHub Actions workflow (`.github/workflows/ci.yml`).
- Code coverage targets are monitored (aim for >80% where practical).


## **4. Example Test Cases**

### **4.1. Unit Tests**

**Test:** Data Loader parses valid ClinicalTrials.gov XML  
- **Given:** A well-formed example XML file  
- **When:** The loader parses the file  
- **Then:** All expected fields (NCT ID, title, population, etc.) are present and correctly populated

**Test:** Text Preprocessing removes HTML tags, normalizes whitespace  
- **Given:** Text with markup and irregular spaces  
- **When:** Preprocess function is applied  
- **Then:** Output is clean, plain text



### **4.2. Integration Tests**

**Test:** Information Extraction Pipeline  
- **Given:** Sample annotated ClinicalTrials.gov file  
- **When:** Run through full extraction pipeline  
- **Then:** Extracted PICO elements match expected ground truth

**Test:** Summarization  
- **Given:** Example study description  
- **When:** Summarization model runs  
- **Then:** Output is non-empty, concise, and contains main study points (qualitative review)



### **4.3. End-to-End (E2E) Tests**

**Test:** User uploads file via Streamlit and downloads results  
- **Given:** Application is running locally  
- **When:** User uploads an example file, triggers extraction, then downloads results  
- **Then:** Result file is created and contains correctly structured data without errors


### **4.4. Regression Tests**

**Test:** Regression on previously fixed annotation bug  
- **Given:** Known edge-case file that triggered a bug in the past  
- **When:** Extraction pipeline is run  
- **Then:** Previous bug should not occur; extraction is correct


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

- Expand test coverage with all new features and bug-fixes
- Code reviews and peer testing for all pull requests
- Regularly review automated test results and resolve any red (failing) tests before release
- Update `/tests/` with any new edge-case files encountered during user feedback

