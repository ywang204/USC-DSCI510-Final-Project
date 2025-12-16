# USC-DSCI510-Final-Project
“Will This Game Be a Hit?”:
Explaining and Predicting Video Game Review Scores Using Machine Learning

## Team Members
1. **Name**: Yuhan Wang
**USC ID**: 8914950372
**Email**: ywang204@usc.edu

2.**Name**: Kaiyue Deng
**USC ID**: [Put Your USC ID Here]
**Email**: [Put Your Email Here]

## Project Description
This project analyzes the relationship between game reviews' textual sentiment and their numerical ratings using VADER sentiment analysis and Linear Regression. 
It scrapes data, cleans the text, performs aspect-based sentiment analysis, audits the model for bias, and visualizes the results.

## Directory Structure
```text
my_project/
├── data/
│   ├── raw/             # Raw JSON/CSV data and attribution keywords
│   └── processed/       # Cleaned and analyzed CSV files
├── results/
│   ├── figures/         # Generated visualizations (PNG)
│   └── regression_coefficients.csv
├── src/
│   ├── get_data.py
│   ├── clean_data.py
│   ├── run_analysis.py
│   ├── visualize_results.py
│   └── regression.py
├── requirements.txt
└── README.md
