TODO: check the conmtent of this file once I make up my mind on this. Check also the hyperlinks!

# **Evaluation Metrics and Methods**

**Project:** Biomedical LLM Information Extraction Tool  
**Version:** 0.1  
**Author:** [Your Name]  
**Date:** [Date]



## **1. Introduction**

This document describes the evaluation strategy for the Biomedical LLM Information Extraction Tool. The tool’s effectiveness, accuracy, and usability are assessed through both quantitative metrics and qualitative feedback, to ensure that extracted information (e.g., PICO elements) and summaries from clinical trial documents are reliable and actionable for real-world users.



## **2. Evaluation Objectives**

- **Extraction Quality:** Measure how accurately the tool extracts key information (Population, Intervention, Comparator, Outcome – PICO) from clinical trial documents versus a human-annotated reference.
- **Summarization Quality:** Assess the coherence, relevance, and conciseness of machine-generated summaries.
- **Performance:** Evaluate system response times for realistic batch sizes.
- **Usability:** Gather user feedback about clarity, workflow, and utility of the interface.



## **3. Evaluation Metrics**

### 3.1. Extraction Accuracy

All metrics are calculated on "partial" overlap, i.e. counting as a match if one is a substring of the other (in either direction).

- **Precision:**  
  Fraction of extracted PICO elements that are correct.
  
  $$ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} $$

- **Recall:**  
  Fraction of relevant PICO elements in the gold standard that the tool correctly extracts.
  
  $$ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} $$
  
- **F1-Score:**  
  Harmonic mean of precision and recall, providing a balanced measure.
  
  $$ F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{(\text{Precision} + \text{Recall})} $$

- **Match Rate:**  
  Proportion of documents for which all PICO elements are correctly extracted (determined as recall=1). 



### 3.2. Summarization Quality

- **ROUGE-N:**  
  Overlap of generated summary n-grams (typically ROUGE-1 and ROUGE-2) with a human-written reference.
- **ROUGE-L:**  
  Longest common subsequence metric; closer to human summary structure.
- **BLEU Score:**  
  Evaluates precision of n-grams, less commonly used but informative for concise outputs.




### 3.3. System Performance

- **Processing Time:**  
  Average/median time to process and extract outputs from a set of clinical trial documents.
- **Batch Throughput:**  
  Number of documents reliably processed per minute on recommended hardware.


### 3.4. Usability & User Experience (to be done)

- **User Satisfaction Survey:**  
  Users rate the ease of use, clarity of interface, interpretability of outputs, and satisfaction with workflow.
- **Task Success Rate:**  
  Fraction of users who successfully complete a standard extraction and download in an observed test session.



## **4. Evaluation Methods**

### 4.1. Gold Standard Annotation

- Manually annotate a small validation set of ClinicalTrials.gov files with ground-truth PICO elements and summaries.
- Use these as the reference against which automatic extraction and summarization are compared.

### 4.2. Automated Evaluation

- Implement scripts to compute Precision, Recall, F1, ROUGE, and BLEU on the annotated test set.
- Use automated timing/logging to assess system performance with varied input sizes.

### 4.3. Human Evaluation (to be done)

- Domain experts and target users review randomly selected outputs for subjective quality and real-world utility.
- Collect feedback via surveys or user interviews.

### 4.4. User Acceptance Testing (UAT) (to be done)

- Conduct scenario-based walkthroughs with representative users (e.g., researchers, medical affairs professionals).
- Record qualitative feedback and success rates.



## **5. Acceptance Criteria**

- Extraction F1-score ≥ 0.8 on reference set.
- Summarization ROUGE-L ≥ 0.5 on reference summaries.
- Median processing time per document ≤ 30 seconds/document.
- ≥ 80% user task success rate and satisfaction score ≥ 4/5 in pilot tests.


## **6. Continuous Improvement**

- Routine re-evaluation on new data and extended user feedback.
- Update models as necessary to close any performance or usability gaps.
- Expand annotated validation dataset over project lifetime.



## **7. References**

- [ROUGE: Recall-Oriented Understudy for Gisting Evaluation](https://aclanthology.org/W04-1013/)
- [BLEU: Bilingual Evaluation Understudy](https://aclanthology.org/P02-1040/)
- [Precision/Recall/F1 Explained (Wikipedia)](https://en.wikipedia.org/wiki/Precision_and_recall)

