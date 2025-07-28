TODO: check document, once DOCKER deployment is done

# Deployment Instructions and Considerations

**Project:** Biomedical LLM Information Extraction Tool  
**Version:** 0.1  
**Author:** [Your Name]  
**Date:** [Date]



## 1. Overview

This document describes the recommended methods for deploying the Biomedical LLM Information Extraction Tool in local, on-premise, and (optionally) cloud environments. The default method uses Docker for containerized, reproducible deployment that ensures consistency across systems.



## 2. Prerequisites

- **Hardware:**  
  - Minimum: 4 CPU cores, 8GB RAM (more recommended for large documents or batch processing)
  - GPU optional but beneficial for faster model inference (ensure Docker setup supports GPU if used)
- **Software:**  
  - [Docker](https://www.docker.com/get-started) (version 20.x or higher recommended)
  - (Optional for non-Docker installs) Python 3.11+, pip, git


## 3. Quick Start: Docker Deployment

### 3.1. Build the Docker Image

In the project root directory, run:
```bash
docker build -t biomed-extractor .
```

### 3.2. Run the Docker Container

```bash
docker run -p 8501:8501 biomed-extractor
```
- The application will be accessible at [http://localhost:8501](http://localhost:8501) in your web browser.

### 3.3. Using Data/Model Volumes (Optional)

To persist uploads/results or use custom models/data:
```bash
docker run -p 8501:8501 \
    -v /your/local/dir:/code/data \
    biomed-extractor
```


## 4. Environment Variables and Configuration

- **Model Selection:** Edit `app/config.py` or set environment variables as needed for alternate model choices.
- **Resource Limits:** Adjust Docker runtime settings (`--cpus`, `--memory`) for larger workloads.
- **GPU Support:**  
    - Use [`nvidia-docker`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU acceleration.
    - Example:  
      ```bash
      docker run --gpus all -p 8501:8501 biomed-extractor
      ```

---

## 5. Alternative: Local (Non-Docker) Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ElenJ/biomed-extractor.git
    cd biomed-extractor
    ```
2. Install Python requirements:
    ```bash
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```
3. Run the app:
    ```bash
    streamlit run app/main.py
    ```



## 6. Cloud/Enterprise Deployment Considerations

- **On-premise Servers:**  
  - Deploy the Docker container on a secure internal server; map to an internal DNS, e.g., `http://biomed-extractor.company.lan:8501`.
- **Cloud Providers:**  
  - Deploy via AWS ECS, Azure Container Instances, Google Cloud Run, or Kubernetes. Consider using managed secrets for any authentication keys.
  - Always ensure patient or sensitive data remains compliant with privacy regulations (HIPAA, GDPR, etc.).



## 7. Security and Privacy Recommendations

- **Always run behind a secure network/firewall** for non-public deployments.
- **Configure HTTPS** in production environments; use reverse proxies (e.g., nginx, traefik) for TLS/SSL.
- **No external API calls are made by the app unless explicitly configured.**  
- **Regularly update Docker images and dependencies** to patch security vulnerabilities.



## 8. Maintenance and Updates

- **Update process:**  
  - Pull the latest version from GitHub
  - Rebuild the Docker image
  - Restart the container
- **Automated testing with CI:**  
  - All commits are tested via GitHub Actions. Ensure tests pass prior to deployment.



## 9. Troubleshooting

- For common issues, consult `docs/troubleshooting.md`.
- Check Docker logs with:
    ```bash
    docker logs <container_id>
    ```
- Resource errors (memory, CPU): Increase Docker limits or try on a larger machine.



## 10. References

- [Docker Documentation](https://docs.docker.com/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/)
- [Huggingface Transformers Deployment](https://huggingface.co/docs/transformers/installation)

