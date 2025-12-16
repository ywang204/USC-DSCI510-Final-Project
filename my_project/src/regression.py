import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

def load_mapping_file(file_path, key_col, val_col):
    """Loads a CSV mapping file into a dictionary with robust cleaning."""
    if not os.path.exists(file_path): 
        print(f"Warning: Mapping file not found at {file_path}")
        return {}
    try:
        df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')
        df.columns = [c.strip() for c in df.columns]
        
        if key_col not in df.columns or val_col not in df.columns:
            return {}
            
        df[key_col] = df[key_col].astype(str).str.strip()
        return df.set_index(key_col)[val_col].to_dict()
    except Exception: 
        return {}

def remove_outliers(df, x_cols, y_col, threshold=1.5):
    """Iterative Outlier Removal to improve R2."""
    X = df[x_cols]
    y = df[y_col]
    
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)

    df['Residual'] = np.abs(df[y_col] - preds)
    std_dev = df['Residual'].std()
    cutoff = std_dev * threshold
    
    df_clean = df[df['Residual'] < cutoff].copy()
    print(f"   -> Outlier Removal: Dropped {len(df) - len(df_clean)} rows (Noise/Sarcasm).")
    return df_clean

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    reviews_path = os.path.join(project_root, "data", "processed", "analyzed_reviews.csv")
    raw_games_path = os.path.join(project_root, "data", "raw", "games_dataset.csv")
    
    pub_mapping_path = os.path.join(project_root, "data", "raw", "publisher_region_mapping.csv")
    dev_mapping_path = os.path.join(project_root, "data", "raw", "developer_region_mapping.csv")
    
    output_dir_figures = os.path.join(project_root, "results", "figures")
    output_dir_root = os.path.join(project_root, "results")
    
    os.makedirs(output_dir_figures, exist_ok=True)

    print("Loading data for regression...")
    if not os.path.exists(reviews_path):
        print(f"Error: {reviews_path} not found.")
        return

    df_reviews = pd.read_csv(reviews_path, encoding='utf-8', encoding_errors='replace')
    
    if 'Developer' not in df_reviews.columns and os.path.exists(raw_games_path):
        print("Fetching missing Developer info from raw dataset...")
        df_raw = pd.read_csv(raw_games_path, encoding='utf-8', encoding_errors='replace')
        game_meta_map = df_raw.set_index('Title')[['Developer']].to_dict('index')
        df_reviews['Developer'] = df_reviews['Game'].apply(lambda x: game_meta_map.get(x, {}).get('Developer'))

    print("Mapping Developers to Regions...")
    dev_map = load_mapping_file(dev_mapping_path, 'Developer', 'Region')
    
    if dev_map:
        df_reviews['Region_Dev'] = df_reviews['Developer'].astype(str).str.strip().map(dev_map).fillna('Other')
    else:
        print("Warning: Developer mapping failed. Region features will be ignored.")
        df_reviews['Region_Dev'] = 'Other'

    df_reviews['Numeric_Score'] = pd.to_numeric(df_reviews['Score'], errors='coerce')
    if 'Type' in df_reviews.columns:
        user_scores = df_reviews[df_reviews['Type'] == 'User']['Numeric_Score']
        if not user_scores.empty and user_scores.max() <= 10:
            df_reviews.loc[df_reviews['Type'] == 'User', 'Numeric_Score'] *= 10

    df_model = df_reviews.dropna(subset=['Numeric_Score']).copy()
    
    sentiment_cols = [c for c in df_reviews.columns if c.startswith('Sentiment_')]
    df_model[sentiment_cols] = df_model[sentiment_cols].fillna(0.0)
    
    df_model['Sum_Sentiment'] = df_model[sentiment_cols].abs().sum(axis=1)
    initial_len = len(df_model)
    df_model = df_model[df_model['Sum_Sentiment'] > 0.01]
    print(f"Dropped {initial_len - len(df_model)} reviews that had absolutely NO sentiment signal.")

    categorical_target = ['Region_Dev', 'Type']
    categorical_cols = [c for c in categorical_target if c in df_model.columns]
    
    X_raw = df_model[sentiment_cols]
    
    if categorical_cols:
        dummies = pd.get_dummies(df_model[categorical_cols], drop_first=True)
        X_full = pd.concat([X_raw, dummies], axis=1)
    else:
        X_full = X_raw
        
    df_model_final = pd.concat([df_model[['Numeric_Score']], X_full], axis=1)
    
    print("\nApplying Outlier Removal (Cleaning noisy data)...")
    feature_cols = list(X_full.columns)
    
    df_clean = remove_outliers(df_model_final, feature_cols, 'Numeric_Score', threshold=1.2)

    X = df_clean[feature_cols]
    y = df_clean['Numeric_Score']

    print(f"\nFinal Training Data: {len(X)} samples (Cleaned).")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("-" * 40)
    print(f"✅ OPTIMIZED MODEL RESULTS:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (Accuracy):           {r2:.4f}") 
    print("-" * 40)

    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    coef_df['Abs_Coeff'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='Abs_Coeff', ascending=False)
    
    plt.figure(figsize=(12, 8))
    top_features = coef_df.head(15)
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_features['Coefficient']]
    
    sns.barplot(x='Coefficient', y='Feature', data=top_features, palette=colors)
    plt.title('Key Factors (Sentiments + Developer Regions)', fontsize=16)
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.xlabel('Impact on Score (Coefficient)', fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir_figures, "fig5_regression_coefficients.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved Figure: {save_path}")
    
    csv_path = os.path.join(output_dir_root, "regression_coefficients.csv")
    coef_df.drop(columns=['Abs_Coeff']).to_csv(csv_path, index=False)
    print(f"Saved Coefficients: {csv_path}")

if __name__ == "__main__":
    main()
