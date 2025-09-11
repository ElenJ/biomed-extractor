# Troubleshooting Guide

**Biomedical LLM Information Extraction Tool**  
**Version:** 0.1  
**Author:** Elena Jolkver
**Date:** 11.09.2025


This guide lists common issues users may encounter and suggested steps to resolve them.



## 1. Application Does Not Start

**Symptoms:**  
- No response after running the Docker/container or `streamlit run` command.  
- Error message: "Port already in use."

**Possible Causes & Solutions:**
- **Port Conflict:**  
  Another application is using port 8501 (default for Streamlit).  
  **Solution:** Stop the other application or run the tool on a different port (e.g., `docker run -p 8600:8501 ...`), then visit `http://localhost:8600`.
- **Docker Not Installed/Running:**  
  Ensure Docker is installed and the Docker daemon is running.
- **Missing Dependencies (non-Docker):**  
  Make sure you’ve run `pip install -r requirements.txt`.



## 2. Cannot Upload Files / File Upload Fails

**Symptoms:**  
- File does not appear after selection/upload.
- Error message: "Unsupported file type" or "File too large."

**Possible Causes & Solutions:**
- **Wrong File Format:**  
  Only ClinicalTrials.gov json and csv plain text files are supported. Double-check file type.
- **File Size Limits:**  
  File too large for browser upload limit (~200MB default). Try splitting data or using smaller batches.
- **Browser Compatibility:**  
  Try using the latest version of Chrome, Firefox, or Edge.



## 3. Extraction or Summarization Fails

**Symptoms:**  
- Error message on screen: “Extraction failed”, “Model not loaded”, or “Summary unavailable”.

**Possible Causes & Solutions:**
- **Insufficient System Resources:**  
  - Try closing other programs to free memory/CPU.
  - Large batches require more RAM—reduce batch size or use a machine with more resources.
- **Model Not Downloaded/Initialized:**  
  Initial build/download of models can take time. Ensure internet access during first run (unless using pre-bundled models).
- **Corrupted or Unsupported Files:**  
  Check that your documents are valid ClinicalTrials.gov files.



## 4. Output Results Are Empty or Incomplete

**Symptoms:**  
- Table shows "N/A", missing fields, or blank summaries.

**Possible Causes & Solutions:**
- File may lack certain PICO elements, interventions, or outcomes.
- The AI model might not recognize poorly formatted or very brief documents. Try with standard-format files.
- Update to the latest tool version if improvements/fixes have been released.



## 5. Download Button Does Not Work

**Symptoms:**  
- Clicking "Download" does nothing or gives an error.

**Possible Causes & Solutions:**
- Ensure extraction ran successfully first—downloads are only enabled after successful processing.
- Check browser popup/download settings.
- Try refreshing the page (F5) or restarting the app.



## 6. "CUDA" or "Torch not found" Errors (GPU Deployments)

**Symptoms:**  
- Errors mentioning CUDA, torch, or missing device.

**Possible Causes & Solutions:**
- The application will fall back to CPU if GPU is unavailable.
- For GPU use, ensure Nvidia drivers, CUDA, and [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) are installed.
- On non-GPU machines, ignore these warnings; performance may be slower.



## 7. Docker-Specific Issues

**Symptoms:**  
- Container exits immediately, network not accessible.

**Possible Causes & Solutions:**
- See error logs via `docker logs <container_id>` for specific messages.
- Permissions issue: run Docker with necessary permissions or as an administrator.
- For volume mounting, verify correct path syntax:  
  - Linux/Mac: `-v /path/on/host:/code/data`
  - Windows: `-v C:\path\on\host:/code/data`



## 8. Unexpected Application Behavior or UI Crash

**Symptoms:**  
- Page freezes, buttons do not respond, or Streamlit reports internal error.

**Possible Causes & Solutions:**
- Try refreshing the page in your browser.
- Restart the Docker container or Streamlit process.
- Check if large data or unexpected file format caused the crash—retry with a small, valid file for debugging.



## 9. Updating or Resetting the Application

- Pull the latest version from GitHub:
  ```bash
  git pull origin main
  docker build -t biomed-extractor .
  docker run -p 8501:8501 biomed-extractor
  ```
- For persistent issues, delete any cached model/data folders and let the tool reinitialize.



## 10. Getting Help

If your issue is not resolved above:

- Review [user_manual.md](user_manual.md) and [requirements.md](requirements.md)
- Search for your error message in the [project repository issues](https://github.com/ElenaJ/biomed-extractor/issues)
- Contact the deployment administrator or project maintainer.

