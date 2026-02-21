
# Economic-Gambling Comment Analysis

This project processes, analyzes, and classifies public comments and attachments related to economic-gambling, focusing on identifying and quantifying references to economic and gambling terms. It leverages NLP techniques (tokenization, lemmatization, word counting) and machine learning model outputs for classification.

---

## File Tree

```
├── convert_to_excel.py
├── count_words.py
├── DIR_CONST.py
├── Get_PDF_text.py
├── requirements.txt
├── classification/
│   ├── classify_individual_or_organization.py
│   ├── classify_individual_or_organization_majority.py
│   ├── classify_pro_against.py
│   ├── classify_pro_against_with_rule.py
│   ├── classify_pro_against_with_rule_majority.py
│   ├── classify_pro_against_majority.py
│   ├── compare_classification.py
│   └── README.md
├── dups/
│   ├── add_dup_cols.py
│   ├── get_dup_comments.py
│   └── README.md
└── DATA/
    ├── comments.csv
    ├── Classification/
    │   ├── comments_with_classification_gemma_pro_against_with_rule.csv
    │   ├── comments_with_classification_gemma_pro_against.csv
    │   ├── comments_with_classification_gemma_who_submit.csv
    │   └── comparison_classification_pro_against.csv
    ├── Classification X Words count/
    │   ├── comparison_gemma_gambling_economic_claude.csv
    │   └── comparison_gemma_gambling_economic_GPT.csv
    ├── Duplicate IDs/
    │   └── duplicate_ids.csv
    ├── PDF files/
    │   ├── lem/
    │   │   └── *.pdf.len.txt
    │   ├── tok/
    │   └── txt_clean/
    ├── Words/
    │   ├── words_claude.csv
    │   ├── words_gpt.csv
    │   └── Raw/
    └── Words count/
        ├── comments_processed_Claude.csv
        ├── comments_processed_GPT.csv
        └── count gt 0/
```

# Economic-Gambling Comment Analysis

This repository processes, analyzes, and classifies public comments and attachments related to economic-gambling, focusing on references to economic and gambling terms. It uses NLP techniques (tokenization, lemmatization, word counting) and large language models for classification.

---

## Repository Structure

### Main Python Scripts
- **Get_PDF_text.py**: Extracts text from PDF files in the data folder.
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
- Main data folder. See [DATA/README.md](DATA/README.md) for detailed structure and file descriptions.
    - **comments.csv**: All public comments and metadata.
    - **Classification/**: Model classification outputs for economic-gambling tasks.
    - **Classification X Words count/**: CSVs comparing model outputs with word count analyses.
    - **Duplicate IDs/**: Duplicate comment ID tracking.
    - **PDF files/**: Processed PDF files and their text representations.
    - **Words/**: Word lists and dictionaries for economic-gambling classification and analysis.
    - **Words count/**: Processed CSVs with word counts and ratios for each comment.

---

## Workflow: How to Reproduce All Outputs

Follow these steps to reproduce all files and results:

### 1. Extract Text from Attachments
- Run `Get_PDF_text.py` to extract text from PDF files in DATA/PDF files/lem.

### 2. Tokenize and Lemmatize Attachments
- Use functions in `count_words.py` to tokenize and lemmatize attachment text files.
    - Run `tokenize_and_save_attachments()` and `lemmatize_tok_files()`.

### 3. Process Comments for Word Counts
- Use `count_words.py` to process comments and attachments for target word categories (economic, gambling, ambiguous).
    - Run `process_comments_csv()` and related functions.
    - Outputs are saved in DATA/Words count/.

### 4. Classify Comments
- Run classification scripts to label comments:
    - `classify_pro_against.py` (PRO/AGAINST/UNCLEAR/NO COMMENT)
    - `classify_pro_against_majority.py` (majority aggregation)
    - Scripts in `classification/` for individual/organization, rule-based, and comparison tasks.
    - Outputs are saved in DATA/Classification/.

### 5. Compare Results
- Use comparison functions in `count_words.py` and scripts in `classification/` to relate model outputs to word count analyses.
    - Outputs are saved in DATA/Classification X Words count/.

### 6. Handle Duplicate Comments
- Use scripts in `dups/` to annotate and extract duplicate comments.

### 7. Convert Outputs to Excel
- Run `convert_to_excel.py` to convert CSV outputs to Excel format for easier review.

---

## Model Details
- **Gemma (google/gemma-3-12b-it)**: Default instruction-tuned model for classification.
- **Other LLMs**: Alternative models can be used for comparison.

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