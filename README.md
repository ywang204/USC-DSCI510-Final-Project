# USC-DSCI510-Final-Project
“Will This Game Be a Hit?”:
Explaining and Predicting Video Game Review Scores Using Machine Learning

## Team Members
1.**Name**: Yuhan Wang
**USC ID**: 8914950372
**Email**: ywang204@usc.edu

2.**Name**: Kaiyue Deng
**USC ID**: 7459633091
**Email**: kaiyuede@usc.edu

## Project Description
This project investigates the correlation between textual sentiment in game reviews and numerical ratings. 
By utilizing VADER sentiment analysis, we extract aspect-based sentiments (e.g., Gameplay, Visuals, Narrative) from user and critic reviews. 
We implemented a Linear Regression model with Outlier Removal to predict game scores based on these sentiment features. 
The findings aim to reveal how specific aspects of a game contribute to its overall success and help identify "coherent" reviews where text aligns with the score.

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
