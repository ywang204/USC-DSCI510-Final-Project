# Game Sentiment Analysis Project

## Team Members
* **Name**: Yuhan Wang
* **USC ID**: 8914950372
* **Email**: ywang204@usc.edu
/n
* **Name**: Kaiyue Deng
* **USC ID**: 7459633091
* **Email**: kaiyuede@usc.edu

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
```

## Setup Instructions
```text
1. Prerequisites
Ensure you have Python 3.8+ installed.

2. Create and Activate Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

Mac/Linux:
python3 -m venv venv
source venv/bin/activate

Windows:
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
Install all required libraries using the provided requirements.txt file:
pip install -r requirements.txt
Note: The scripts will automatically check for and download necessary NLTK data (punkt, stopwords, vader_lexicon).

How to Run the Project
Please run the scripts in the following order to replicate the full analysis pipeline:

Step 1: Data Collection
Fetches raw game data and saves it to data/raw/.
python src/get_data.py

Step 2: Data Cleaning
Processes raw text, removes noise, and saves structured data to data/processed/.
python src/clean_data.py

Step 3: Sentiment Analysis
Runs VADER sentiment analysis on specific aspects (Gameplay, Visuals, etc.) and audits the data.
python src/run_analysis.py

Step 4: Visualization
Generates plots comparing User vs. Critic scores and sentiment distributions. Figures are saved to results/figures/.
python src/visualize_results.py

Step 5: Regression Modeling
Runs the regression model to predict scores based on sentiment features. It prints the RMSE and R-squared values to the console.
python src/regression.py
```
