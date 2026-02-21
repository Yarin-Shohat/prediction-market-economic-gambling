# DATA Folder Overview

This folder contains all the data files and subfolders used for classification, analysis, and processing of public comments and related documents for the economic-gambling project.

## Folder Structure

### comments.csv
- The main CSV file containing all public comments and metadata.

### Classification/
- Contains classification results and comparison files for economic-gambling classification tasks.
    - `comments_with_classification_gemma_pro_against_with_rule.csv`: Classification results (Gemma model, with rule text).
    - `comments_with_classification_gemma_pro_against.csv`: Classification results (Gemma model, without rule text).
    - `comments_with_classification_gemma_who_submit.csv`: Classification of submitter (individual/organization).
    - `comparison_classification_pro_against.csv`: Comparison of classification results (with/without rule).

### Classification X Words count/
- Contains files comparing classification results with word count statistics for economic-gambling analysis.
    - `comparison_gemma_gambling_economic_claude.csv`: Comparison between Gemma and Claude models (economic-gambling).
    - `comparison_gemma_gambling_economic_GPT.csv`: Comparison between Gemma and GPT models (economic-gambling).

### Duplicate IDs/
- Contains files related to duplicate comment IDs.
    - `duplicate_ids.csv`: List of duplicate comment IDs for analysis.

### PDF files/
- Contains processed PDF files and their text representations.
    - `lem/`: Contains `.pdf.len.txt` files (length statistics for each PDF).
    - `tok/`: Contains tokenized text files from PDFs.
    - `txt_clean/`: Contains cleaned text files from PDFs.

### Words/
- Contains word lists and dictionaries used for economic-gambling classification and analysis.
    - `words_gpt.csv`: Word list from GPT model (economic-gambling).
    - `words_claude.csv`: Word list from Claude model (economic-gambling).
    - `Raw/`: Contains raw word files and data.

### Words count/
- Contains processed comment files with word count statistics for economic-gambling analysis.
    - `comments_processed_Claude.csv`: Processed comments (Claude model).
    - `comments_processed_GPT.csv`: Processed comments (GPT model).
    - `count gt 0/`: Contains files with word count greater than zero.
