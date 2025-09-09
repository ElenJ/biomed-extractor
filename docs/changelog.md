# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/).


## [Unreleased]
### Ideas
- Support for alternative input sources (e.g., PubMed articles, CSV uploads).
- Option for user-selectable NLP/LLM models and on-the-fly model updates.
- Advanced summarization capabilities (multi-document summarization, custom summary lengths).
- User-driven annotation and manual correction interface for extracted elements.
- Integrated Q&A (Question Answering) functionality over uploaded texts.
- Workflow history and activity logging for user sessions.
- Enhanced access management for multi-user deployments.
- Docker volume support for persistent user data and results.
- CLI interface for headless/batch processing.
- summarization via LLM-model
- Improved UI customization and themes.
- Streamlined error handling and feedback messages.

### Changed
- xxx

### Fixed
- (To be updated upon identification of issues after first release.)



## [0.1.0] - 2025-10-01
### Added
- Initial Streamlit UI for uploading ClinicalTrials.gov files and visualizing extraction results.
- Modular NLP pipeline for PICO element extraction using bio-mobilebert or BioELECTRA.
- Extractive summarization using TextRank algorithm.
- File processing and results download (CSV).
- Dockerfile for containerized deployment.
- Basic unit and integration tests.
- User documentation: manual, architecture, requirements, troubleshooting.

