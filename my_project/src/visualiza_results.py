import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def load_mapping_file(file_path, key_col, val_col):
    if not os.path.exists(file_path):
        print(f"Warning: Mapping file not found at {file_path}")
        return {}
    try:
        df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')
        if key_col not in df.columns or val_col not in df.columns:
            print(f"Error: Columns {key_col}/{val_col} missing in {file_path}")
            return {}
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
    raw_games_path = os.path.join(project_root, "data", "raw", "games_dataset.csv")
    
    pub_mapping_path = os.path.join(project_root, "data","processed", "publisher_region_mapping.csv")
    dev_mapping_path = os.path.join(project_root, "data", "processed","developer_region_mapping.csv")

    print(f"Output Directory set to: {output_dir}")

    print("Loading datasets...")
    if not os.path.exists(reviews_path):
        print(f"Error: Reviews file not found at {reviews_path}. Please run run_analysis.py first.")
        return

    try:
        df_reviews = pd.read_csv(reviews_path, encoding='utf-8', encoding_errors='replace')
        if os.path.exists(raw_games_path):
            df_raw = pd.read_csv(raw_games_path, encoding='utf-8', encoding_errors='replace')
        else:
            df_raw = pd.DataFrame()
            print("Warning: Raw games dataset not found. Some metadata might be missing.")
            
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    print("Integrating metadata...")
    
    if not df_raw.empty:
        game_meta_map = df_raw.set_index('Title')[['Publisher', 'Developer']].to_dict('index')
        
        if 'Publisher' not in df_reviews.columns:
            df_reviews['Publisher'] = df_reviews['Game'].apply(lambda x: game_meta_map.get(x, {}).get('Publisher'))
        if 'Developer' not in df_reviews.columns:
            df_reviews['Developer'] = df_reviews['Game'].apply(lambda x: game_meta_map.get(x, {}).get('Developer'))

    pub_region_map = load_mapping_file(pub_mapping_path, 'Publisher', 'Region')
    dev_region_map = load_mapping_file(dev_mapping_path, 'Developer', 'Region')
    
    df_reviews['Region_Pub'] = df_reviews['Publisher'].map(pub_region_map).fillna('Unknown')
    df_reviews['Region_Dev'] = df_reviews['Developer'].map(dev_region_map).fillna('Unknown')

    df_reviews['Numeric_Score'] = pd.to_numeric(df_reviews['Score'], errors='coerce')
    
    # Normalize User Scores (usually 0-10) to 0-100 scale to match Critics
    if 'Type' in df_reviews.columns:
        user_scores = df_reviews[df_reviews['Type'] == 'User']['Numeric_Score']
        if not user_scores.empty and user_scores.max() <= 10:
            df_reviews.loc[df_reviews['Type'] == 'User', 'Numeric_Score'] *= 10
    
    df_clean = df_reviews.dropna(subset=['Numeric_Score'])

    sns.set_theme(style="whitegrid", context="notebook")

    # FIGURE 1A: Publisher Region vs Ratings
    print("Generating Figure 1A (Publisher Region)...")
    plt.figure(figsize=(12, 7))
    
    valid_counts = df_clean['Region_Pub'].value_counts()
    major_regions = valid_counts[valid_counts > 5].index.tolist()
    if 'Unknown' in major_regions: major_regions.remove('Unknown')
    if 'Multi-region' in major_regions: major_regions.remove('Multi-region')
    
    if major_regions:
        plot_data = df_clean[df_clean['Region_Pub'].isin(major_regions)].copy()
        order = plot_data.groupby('Region_Pub')['Numeric_Score'].median().sort_values(ascending=False).index
        
        counts = plot_data['Region_Pub'].value_counts()
        labels = [f"{r}\n(n={counts[r]})" for r in order]
        
        ax = sns.boxplot(x='Region_Pub', y='Numeric_Score', data=plot_data, order=order, palette='Spectral', showfliers=False)
        sns.stripplot(x='Region_Pub', y='Numeric_Score', data=plot_data, order=order, color='black', alpha=0.3, size=3)
        
        ax.set_xticklabels(labels)
        plt.title('Game Ratings by Publisher Region', fontsize=16)
        plt.ylabel('Score (0-100)')
        plt.xlabel('')
        plt.ylim(0, 105)
        
        save_path = os.path.join(output_dir, "fig1a_publisher_region.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")
    else:
        print("Skipping Figure 1A: Not enough regional data.")

    # FIGURE 1B: Developer Region vs Ratings
    print("Generating Figure 1B (Developer Region)...")
    plt.figure(figsize=(12, 7))
    
    valid_counts_dev = df_clean['Region_Dev'].value_counts()
    major_regions_dev = valid_counts_dev[valid_counts_dev > 5].index.tolist()
    filter_out = ['Unknown', 'Multi-region', 'Other']
    major_regions_dev = [r for r in major_regions_dev if r not in filter_out]

    plot_data_dev = df_clean[df_clean['Region_Dev'].isin(major_regions_dev)].copy()
    
    if not plot_data_dev.empty:
        order_dev = plot_data_dev.groupby('Region_Dev')['Numeric_Score'].median().sort_values(ascending=False).index
        
        counts_dev = plot_data_dev['Region_Dev'].value_counts()
        labels_dev = [f"{r}\n(n={counts_dev[r]})" for r in order_dev]
        
        ax = sns.boxplot(x='Region_Dev', y='Numeric_Score', data=plot_data_dev, order=order_dev, palette='coolwarm', showfliers=False)
        sns.stripplot(x='Region_Dev', y='Numeric_Score', data=plot_data_dev, order=order_dev, color='black', alpha=0.3, size=3)
        
        ax.set_xticklabels(labels_dev)
        plt.title('Game Ratings by Developer Region (Detailed Analysis)', fontsize=16)
        plt.ylabel('Score (0-100)')
        plt.xlabel('')
        plt.ylim(0, 105)
        
        save_path = os.path.join(output_dir, "fig1b_developer_region.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")
    else:
        print("Skipping Figure 1B: Not enough developer data matched.")

    # FIGURE 2: Aspect Influence (Correlation Bar Plot)
    print("Generating Figure 2 (Aspect Correlation)...")
    aspect_cols = [c for c in df_clean.columns if c.startswith('Sentiment_')]
    
    top_aspect_col = None

    if aspect_cols:
        correlations = {}
        for col in aspect_cols:
            valid_rows = df_clean[['Numeric_Score', col]].dropna()
            if not valid_rows.empty:
                corr = valid_rows['Numeric_Score'].corr(valid_rows[col])
                label = col.replace('Sentiment_', '').replace('_', ' ').title()
                correlations[label] = corr
            
        if correlations:
            corr_df = pd.DataFrame(list(correlations.items()), columns=['Aspect', 'Correlation'])
            corr_df = corr_df.sort_values(by='Correlation', ascending=False)
            
            top_aspect_name = corr_df.iloc[0]['Aspect']
            top_aspect_col = "Sentiment_" + top_aspect_name.replace(' ', '_').lower()
            if top_aspect_col not in df_clean.columns:
                 for c in aspect_cols:
                     if top_aspect_name.lower() in c.lower():
                         top_aspect_col = c
                         break

            colors = ['#e74c3c' if x > 0 else '#3498db' for x in corr_df['Correlation']]
            
            plt.figure(figsize=(16, 8))
            sns.barplot(x='Correlation', y='Aspect', data=corr_df, palette=colors)
            plt.title('Correlation: Game Aspects vs. Final Rating', fontsize=16)
            plt.axvline(0, color='black', linewidth=0.8)
            
            save_path = os.path.join(output_dir, "fig2_aspect_influence.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Saved: {save_path}")

    # FIGURE 3: Critics vs Users (Grouped Bar Plot)
    print("Generating Figure 3 (Critic vs User)...")
    if aspect_cols:
        melted = df_clean.melt(id_vars=['Type'], value_vars=aspect_cols, var_name='Raw', value_name='Sentiment')
        melted['Aspect'] = melted['Raw'].str.replace('Sentiment_', '').str.replace('_', ' ').str.title()
        
        plt.figure(figsize=(14, 12))
        sns.barplot(x='Aspect', y='Sentiment', hue='Type', data=melted, palette='Set2', errorbar=None)
        plt.title('Critics vs. Users: Sentiment Focus', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        
        save_path = os.path.join(output_dir, "fig3_critic_vs_user.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    # FIGURE 4: Top Aspect Scatter Plot
    print("Generating Figure 4 (Scatter Plot)...")
    if aspect_cols and top_aspect_col and top_aspect_col in df_clean.columns:
        plt.figure(figsize=(10, 6))
        plot_data = df_clean.sample(min(1000, len(df_clean)), random_state=42) 
        sns.regplot(
            x=top_aspect_col, y='Numeric_Score', data=plot_data, 
            scatter_kws={'alpha':0.5, 'color': '#2ecc71'}, line_kws={'color': 'red'}
        )
        clean_name = top_aspect_col.replace('Sentiment_', '').replace('_', ' ').title()
        plt.title(f'Scatter Plot: {clean_name} Sentiment vs. Rating', fontsize=16)
        
        save_path = os.path.join(output_dir, "fig4_scatter_top_aspect.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    print("\nAll visualizations complete! Check 'results/figures'.")

if __name__ == "__main__":
    main()