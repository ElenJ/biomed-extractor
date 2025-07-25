# biomed-extractor
Biomedical Document Assistant: An LLM-Powered Information Extraction and Summarization Tool for Clinical Studies


Automatically extract PICO elements & structured summaries from ClinicalTrials.gov data.

## Features
- Upload clinical trial data
- Extract population/intervention/outcome
- Summarize key findings via LLM
- Web-based interface (Streamlit)
- Ready for Docker deployment

## Quickstart

```bash
docker build -t biomed-extractor .
docker run -p 8501:8501 biomed-extractor
```
