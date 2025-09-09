# **User Interface Design and User Experience Considerations**

**Project:** Biomedical LLM Information Extraction Tool  
**Version:** 0.1  
**Author:** Elena Jolkver
**Date:** 25.07.2025


## **1. Overview**

The user interface (UI) is implemented using Streamlit, designed to enable domain experts—such as clinical researchers, medical affairs professionals, and data scientists—to efficiently upload clinical trial documents, configure extraction settings, visualize results, and download structured outputs. The UI prioritizes simplicity, clarity, and accessibility, allowing users with minimal technical skill to operate the tool with confidence.



## **2. Design Principles**

- **Simplicity:**  
  Minimal controls and clutter; clear wording; default options for fast, typical usage.

- **Guided Workflow:**  
  The interface follows a logical, step-by-step process: Upload → Configure → Run Extraction → Review & Download.

- **Feedback and Affordance:**  
  Immediate visual feedback for user actions (e.g., file upload confirmations, loading spinners, error/success messages).

- **Accessibility:**  
  Large, readable fonts; high-contrast color scheme; clear labels; keyboard accessibility where feasible.

- **Consistency:**  
  Uniform placement and styling of buttons, uploaders, tables, and messages. Consistent color and icon usage throughout.



## **3. Main User Flow**

### **3.1. File Upload**

- Users upload one or multiple ClinicalTrials.gov files (XML or text) via a File Uploader component at the top of the page.
- Only supported file types are accepted; unsupported formats are blocked with clear error messaging.

### **3.2. Configuration Options**

- Optional: Model selection dropdown, batch size limits, extraction options (e.g., select which PICO elements to extract).
- Tooltips and documentation links provided for all advanced configuration fields.

### **3.3. Extraction Trigger**

- Processing is initislized once data is provided and model is selected
- prominent “Run Extraction” button initiates processing.
- Loading spinner and progress bar reassure users during processing.
- If multiple files, show progress as a percentage or per-document ticker.

### **3.4. Results Display**

- Extracted information presented in an interactive, scrollable table (e.g., one row per study, with columns for Population, Intervention, Comparator, Outcomes, Summary).
- Option to expand/collapse summary details for each file.
- Color highlights or icons to flag missing data or extraction errors.

### **3.5. Download**

- Results downloadable in CSV format via clearly labeled download buttons.
- Download buttons are deactivated/greyed-out until extraction is complete.

### **3.6. Error Handling and Help**

- User-friendly error messages for common issues (e.g., no file uploaded, wrong format, processing failures).
- Link to user documentation and troubleshooting guide.
- Optionally, “Contact Support” or “Open Issue” link for feedback.



## **4. Navigation and Layout**

- **Single Page:**  
  The UI is organized as a single-scroll page to reduce navigation friction.
- **Sectioning:**  
  Visual dividers or expandable sections separate upload, configuration, results, and help.



## **5. Visual Style**

- **Color Scheme:**  
  Professional, high-contrast; accent color for primary actions (e.g., blue or green for “Run Extraction”, red for errors).
- **Consistent Elements:**  
  Buttons, tables, and alerts styled consistently.
- **Responsive Design:**  
  Layout adapts for desktop and tablet use; mobile compatibility is a future enhancement.



## **6. Accessibility and Internationalization**

- Alt text for all icons/images.
- All controls labeled for screen readers.
- Potential for future multi-language support.

## **7. Example Screenshots/Wireframes**

TODO: Include PNG or draw.io wireframes here `design_diagram.png`. 


## **8. Future UI Enhancements**

- Wizard-like stepper for multi-stage workflows.
- Inline correction/annotation (user can edit extracted elements before download).
- User-customizable color themes and layouts.
- Keyboard navigation and more comprehensive a11y support.
- Activity log/history for auditing.

## **9. References**

- Streamlit documentation and best practices
- User feedback (to be collected)
- UX resources on data tool design

