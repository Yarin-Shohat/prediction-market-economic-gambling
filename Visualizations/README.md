# Visualizations Folder

This folder contains scripts, notebooks, and plots for visualizing the results of the economic-gambling comment analysis project.

## Purpose
The purpose of this folder is to provide graphical and statistical insights into the processed and classified public comments related to prediction markets, economic theory, and gambling. Visualizations help interpret model outputs, word counts, and classification results, making the analysis accessible and actionable.

## Contents
- **Visualize_Prediction_Market.ipynb**: Main Jupyter notebook for generating plots and visual summaries. It loads processed data and classification results to create visualizations.
- **plots/**: Directory containing generated plot images and figures.
- **null_id_no_comment_rows.csv**: CSV file tracking comments with missing IDs or no comment text, used for data quality checks.
- **README.md**: This documentation file.

## Data Sources Used
Plots and analyses in this folder use the following CSV files from other parts of the repository:
- `comments_with_classification_gemma_pro_against.csv` (from DATA/Classification/): Contains model-based classification of comments (PRO/AGAINST/UNCLEAR/NO COMMENT).
- `comments_processed_Claude.csv` (from DATA/Words count/): Contains word counts and ratios for each comment, processed using Claude word lists.
- `comments_processed_GPT.csv` (from DATA/Words count/): Contains word counts and ratios for each comment, processed using GPT word lists.

These files are loaded in the notebook as:
- `classification = pd.read_csv('/content/comments_with_classification_gemma_pro_against.csv')`
- `claude_words = pd.read_csv('/content/comments_processed_Claude.csv')`
- `gpt_words = pd.read_csv('/content/comments_processed_GPT.csv')`

## Typical Workflow
1. Run the notebook (`Visualize_Prediction_Market.ipynb`) to generate plots and summary statistics.
2. Plots are saved in the `plots/` directory for review and inclusion in reports.
3. Use the visualizations to compare classification results, word count distributions, and relationships between comment content and model outputs.

## Notes
- All source CSVs are taken from the main data and classification folders; this folder does not modify the original data.
- Visualizations are modular and can be extended for additional analyses.
- For more details, see the notebook and comments within the plotting scripts.
