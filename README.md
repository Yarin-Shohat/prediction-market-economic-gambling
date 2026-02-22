

# Economic-Gambling Comment Analysis

This project processes, analyzes, and classifies public comments and attachments related to economic-gambling, focusing on identifying and quantifying references to economic and gambling terms. It leverages NLP techniques (tokenization, lemmatization, word counting) and machine learning model outputs for classification.

> **Related Project:** For analysis of insider trading-related words in prediction market comments, see the [prediction-market-insider-trading](https://github.com/Yarin-Shohat/prediction-market-insider-trading) repository, which applies similar NLP and classification techniques to analyze different word categories.


---

## Workflow Overview

The analysis pipeline consists of several phases:

### **Phase 0: Data Collection**
- **Folder:** `Data Collection/`
- **Purpose:** Automatically scrapes and downloads all public comments and attachments from the CFTC website for a specific rulemaking (ID 7512).
- **Key Script:** `main.py` (see [`Data Collection/README.md`](Data%20Collection/README.md) for full details)
- **Outputs:**
    - `comments.csv`: All raw comment data and metadata
    - `attachments/`: All downloaded files referenced in the comments
- **How to Run:**
    1. Install dependencies from `requirements.txt` (preferably in a virtual environment)
    2. Run `python main.py` inside the `Data Collection` folder
    3. See [`Data Collection/README.md`](Data%20Collection/README.md) for Docker usage and troubleshooting

### **Phase 1: Extract Text from Attachments**
- Run `Get_PDF_text.py` to extract text from PDF/DOCX files in `DATA/PDF files/Raw`.

### **Phase 2: Tokenize and Lemmatize Attachments**
- Use functions in `count_words.py` to tokenize and lemmatize attachment text files.
    - Run `tokenize_and_save_attachments()` and `lemmatize_tok_files()`.

### **Phase 3: Process Comments for Word Counts**
- Use `count_words.py` to process comments and attachments for target word categories (gambling, economic, ambiguous).
    - Run `process_comments_csv()` and related functions.
    - Outputs are saved in `DATA/Words count/`.

### **Phase 4: Classify Comments**
- Run classification scripts to label comments:
    - `classification/classify_pro_against.py` (PRO/AGAINST/UNCLEAR/NO COMMENT)
    - `classification/classify_pro_against_majority.py` (majority aggregation)
    - Scripts in `classification/` for individual/organization, rule-based, and comparison tasks.
    - Outputs are saved in `DATA/Classification/`.

### **Phase 5: Compare Results**
- Use comparison functions in `count_words.py` and scripts in `classification/` to relate model outputs to word count analyses.
    - Outputs are saved in `DATA/Classification X Words count/`.

### **Phase 6: Handle Duplicate and Missed Comments**
- Use scripts in `dups/` to annotate and extract duplicate comments.
- Use `handle_missed_comments()` and `fill_missed_comments_all()` in `count_words.py` to ensure all comments are processed.

### **Phase 7: Convert Outputs to Excel**
- Run `convert_to_excel.py` to convert CSV outputs to Excel format for easier review.

---

## Repository Structure

### Main Python Scripts
- **Data Collection/**: Automated scraping and download of all public comments and attachments. See its [`README`](Data%20Collection/README.md) for details.
- **Get_PDF_text.py**: Extracts text from PDF/DOCX files in the data folder.
- **count_words.py**: Tokenizes, lemmatizes, and counts target words in comments and attachments. Also compares word counts with model outputs.
- **convert_to_excel.py**: Converts CSV outputs to Excel format for easier review.
- **DIR_CONST.py**: Defines directory constants used throughout the project.

### classification/
- Contains scripts for various classification tasks:
    - `classify_individual_or_organization.py`, `classify_individual_or_organization_majority.py`: Classify submitters as individuals or organizations.
    - `classify_pro_against.py`, `classify_pro_against_with_rule.py`, `classify_pro_against_with_rule_majority.py`, `classify_pro_against_majority.py`: Classify comments as pro/against economic-gambling, with or without rule text, and with/without majority logic.
    - `compare_classification.py`: Compare different classification results.
    - `README.md`: Documentation for classification scripts.

### dups/
- Scripts and data for handling duplicate IDs in comments.
    - `add_dup_cols.py`: Add duplicate columns to data.
    - `get_dup_comments.py`: Extract duplicate comments.
    - `README.md`: Documentation for duplicate handling.

### DATA/
- Main data folder. See [`DATA/README.md`](DATA/README.md) for detailed structure and file descriptions.
    - **comments.csv**: All public comments and metadata.
    - **Classification/**: Model classification outputs for economic-gambling tasks.
    - **Classification X Words count/**: CSVs comparing model outputs with word count analyses.
    - **Duplicate IDs/**: Duplicate comment ID tracking.
    - **PDF files/**: Processed PDF files and their text representations.
    - **Words/**: Word lists and dictionaries for economic-gambling classification and analysis.
    - **Words count/**: Processed CSVs with word counts and ratios for each comment.

---

## Model Details
- **Gemma (google/gemma-3-12b-it)**: Default instruction-tuned model for classification.
- **Llama3 (meta-llama/Llama-3.1-8B)**: Alternative model, selectable via command-line argument.

---

## Reproducibility Tips
- Use the same input files and parameters for consistent results.
- Ensure the same model version and configuration.
- Run scripts in the order above for full pipeline.
- All file paths are hardcoded; modify DIR_CONST.py if you need to change locations.

---

## Additional Notes
- All processing is modular; you can run only the steps you need.
- Word lists can be updated in the Words directory.
- Output CSVs are saved in DATA/Words count/ and DATA/Classification X Words count/.
- For more details, see the docstrings in each Python file and the README files in each subfolder.