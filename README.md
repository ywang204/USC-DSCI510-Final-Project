# Game Sentiment Analysis Project

## Team Members
* **Name**: [Your Name]
* **USC ID**: [Your USC ID]
* **Email**: [Your Email]

## Project Description
This project investigates the correlation between **textual sentiment in game reviews** and **numerical ratings**. 
By utilizing **VADER sentiment analysis**, we extract aspect-based sentiments (e.g., Gameplay, Visuals, Narrative) from user and critic reviews. 
We implemented a **Linear Regression model with Outlier Removal** to predict game scores based on these sentiment features. 
The findings aim to reveal how specific aspects of a game contribute to its overall success.

## Directory Structure
```text
github_repo_structure/
├── data/
│   ├── raw/             # Raw data and attribution keywords
│   └── processed/       # Cleaned and analyzed CSV files
├── results/
│   ├── final_report.pdf # The final project report
│   ├── regression_coefficients.csv
│   └── figures/         # Generated visualizations
├── src/
│   ├── get_data.py
│   ├── clean_data.py
│   ├── run_analysis.py
│   ├── visualize_results.py
│   ├── regression.py    # (Crucial) The outlier removal regression model
│   └── utils/
├── requirements.txt     # Python dependencies
├── project_proposal.pdf
└── README.md
