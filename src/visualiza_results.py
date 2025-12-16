import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def load_mapping_file(file_path, key_col, val_col):
    """Loads a CSV mapping file into a dictionary with robust error handling."""
    if not os.path.exists(file_path):
        print(f"Warning: Mapping file not found at {file_path}")
        return {}
    try:
        df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')
        df.columns = [c.strip() for c in df.columns]
        
        if key_col not in df.columns or val_col not in df.columns:
            print(f"Error: Columns {key_col}/{val_col} missing in {file_path}")
            print(f"Found columns: {list(df.columns)}")
            return {}
            
        df[key_col] = df[key_col].astype(str).str.strip()
        return df.set_index(key_col)[val_col].to_dict()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    output_dir = os.path.join(project_root, "results", "figures")
    os.makedirs(output_dir, exist_ok=True)

    reviews_path = os.path.join(project_root, "data", "processed", "analyzed_reviews.csv")
    
    pub_mapping_path = os.path.join(project_root, "data", "raw", "publisher_region_mapping.csv")
    dev_mapping_path = os.path.join(project_root, "data", "raw", "developer_region_mapping.csv")

    print(f"Output Directory: {output_dir}")

    if not os.path.exists(reviews_path):
        print(f"Error: {reviews_path} not found. Please run run_analysis.py first.")
        return

    print("Loading datasets...")
    df = pd.read_csv(reviews_path)

    def clean_score(row):
        try:
            val = float(row['Score'])
            if row['Type'] == 'User' and val <= 10: 
                return val * 10
            return val
        except:
            return None

    df['Numeric_Score'] = df.apply(clean_score, axis=1)
    df_clean = df.dropna(subset=['Numeric_Score']).copy()

    print("Loading region mappings...")
    pub_map = load_mapping_file(pub_mapping_path, "Publisher", "Region")
    dev_map = load_mapping_file(dev_mapping_path, "Developer", "Region")

    sns.set_theme(style="whitegrid", context="notebook")

    # --- FIGURE 1A: Publisher Region ---
    print("Generating Figure 1A (Publisher Region)...")
    if pub_map and 'Publisher' in df_clean.columns:
        df_clean['Region_Pub'] = df_clean['Publisher'].astype(str).str.strip().map(pub_map)
        
        plot_data = df_clean.dropna(subset=['Region_Pub'])
        plot_data = plot_data[~plot_data['Region_Pub'].isin(['Multi-region', 'Unknown', 'Other'])]

        if not plot_data.empty:
            plt.figure(figsize=(12, 7))
            order = plot_data.groupby('Region_Pub')['Numeric_Score'].median().sort_values(ascending=False).index
            sns.boxplot(x='Region_Pub', y='Numeric_Score', data=plot_data, order=order, palette='Spectral', showfliers=False)
            plt.title('Game Ratings by Publisher Region')
            plt.ylabel('Score (0-100)')
            plt.savefig(os.path.join(output_dir, "fig1a_publisher_region.png"), dpi=300)
            plt.close()
            print("Saved Figure 1A.")
        else:
            print("Skipping Figure 1A: Not enough valid regional data.")
    else:
        print("Skipping Figure 1A: Mapping failed or column missing.")

    # --- FIGURE 1B: Developer Region ---
    print("Generating Figure 1B (Developer Region)...")
    if dev_map and 'Developer' in df_clean.columns:
        df_clean['Region_Dev'] = df_clean['Developer'].astype(str).str.strip().map(dev_map)
        
        plot_data = df_clean.dropna(subset=['Region_Dev'])
        plot_data = plot_data[~plot_data['Region_Dev'].isin(['Multi-region', 'Unknown', 'Other'])]

        if not plot_data.empty:
            plt.figure(figsize=(12, 7))
            order = plot_data.groupby('Region_Dev')['Numeric_Score'].median().sort_values(ascending=False).index
            sns.boxplot(x='Region_Dev', y='Numeric_Score', data=plot_data, order=order, palette='coolwarm', showfliers=False)
            plt.title('Game Ratings by Developer Region')
            plt.ylabel('Score (0-100)')
            plt.savefig(os.path.join(output_dir, "fig1b_developer_region.png"), dpi=300)
            plt.close()
            print("Saved Figure 1B.")
        else:
            print("Skipping Figure 1B: Not enough valid regional data.")
    else:
        print("Skipping Figure 1B: Mapping failed or column missing.")

    # --- FIGURE 2: Aspect Correlation---
    print("Generating Figure 2 (Correlation)...")
    sentiment_cols = [c for c in df_clean.columns if c.startswith('Sentiment_')]
    top_aspect_col = None

    if sentiment_cols:
        correlations = {}
        for col in sentiment_cols:
            valid_rows = df_clean[['Numeric_Score', col]].dropna()
            if not valid_rows.empty:
                corr = valid_rows['Numeric_Score'].corr(valid_rows[col])
                correlations[col] = corr
        
        if correlations:
            corr_df = pd.DataFrame(list(correlations.items()), columns=['Col', 'Correlation'])
            corr_df['Aspect'] = corr_df['Col'].str.replace('Sentiment_', '').str.replace('_', ' ').str.title()
            corr_df = corr_df.sort_values(by='Correlation', ascending=False)
            
            top_aspect_col = corr_df.iloc[0]['Col']
            
            plt.figure(figsize=(10, 8))
            colors = ['#e74c3c' if x > 0 else '#3498db' for x in corr_df['Correlation']]
            sns.barplot(x='Correlation', y='Aspect', data=corr_df, palette=colors)
            plt.title('Correlation: Sentiment Aspects vs Score')
            plt.axvline(0, color='black', linewidth=0.8)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "fig2_aspect_influence.png"), dpi=300)
            plt.close()
            print("Saved Figure 2.")

    # --- FIGURE 3: User vs Critic ---
    print("Generating Figure 3 (User vs Critic)...")
    if sentiment_cols:
        melted = df_clean.melt(id_vars=['Type'], value_vars=sentiment_cols, var_name='Raw', value_name='Sentiment')
        melted['Aspect'] = melted['Raw'].str.replace('Sentiment_', '').str.replace('_', ' ').str.title()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Aspect', y='Sentiment', hue='Type', data=melted, palette='Set2', errorbar=None)
        plt.title('Sentiment Intensity: Users vs Critics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig3_critic_vs_user.png"), dpi=300)
        plt.close()
        print("Saved Figure 3.")

    # --- FIGURE 4: Top Aspect Scatter Plot ---
    print("Generating Figure 4 (Scatter Plot)...")
    if top_aspect_col and top_aspect_col in df_clean.columns:
        plt.figure(figsize=(10, 6))
        sample_n = min(1000, len(df_clean))
        plot_data = df_clean.sample(sample_n, random_state=42)
        
        sns.regplot(
            x=top_aspect_col, y='Numeric_Score', data=plot_data, 
            scatter_kws={'alpha':0.5, 'color': '#2ecc71'}, 
            line_kws={'color': 'red'}
        )
        
        clean_name = top_aspect_col.replace('Sentiment_', '').replace('_', ' ').title()
        plt.title(f'Scatter Plot: {clean_name} Sentiment vs. Rating', fontsize=16)
        plt.xlabel(f'{clean_name} Sentiment Score')
        plt.ylabel('Game Rating (0-100)')
        
        save_path = os.path.join(output_dir, "fig4_scatter_top_aspect.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved Figure 4: {save_path}")
    else:
        print("Skipping Figure 4: No top aspect identified.")

    print("\nAll visualizations complete! Check 'results/figures'.")

if __name__ == "__main__":
    main()