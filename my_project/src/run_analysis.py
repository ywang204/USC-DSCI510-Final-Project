import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import os
from collections import Counter
import string

print("Checking NLTK resources...")
resources = ['vader_lexicon', 'punkt', 'punkt_tab', 'stopwords']
for r in resources:
    try:
        if r == 'vader_lexicon':
            nltk.data.find('sentiment/vader_lexicon.zip')
        elif r == 'stopwords':
            nltk.data.find('corpora/stopwords.zip')
        else:
            nltk.data.find(f'tokenizers/{r}')
    except LookupError:
        nltk.download(r, quiet=True)

def get_culprit_words(text_series, sia, target_type="negative"):
    stop_words = set(stopwords.words('english'))
    culprit_words = []
    
    for text in text_series:
        if not isinstance(text, str): continue
        words = word_tokenize(text.lower())
        for word in words:
            if word in stop_words or word in string.punctuation: continue
            
            score = sia.lexicon.get(word, 0)
            
            if target_type == "negative":
                # False Negatives
                if score < -0.5:
                    culprit_words.append((word, score))
            elif target_type == "positive":
                # False Positives
                if score > 0.5:
                    culprit_words.append((word, score))
                    
    return culprit_words

def print_diagnostic_table(title, culprit_list, limit=20):
    counts = Counter(culprit_list)
    print(f"\n{title}")
    print(f"{'Word':<15} | {'VADER Score':<12} | {'Frequency':<10}")
    print("-" * 45)
    
    for (word, score), count in counts.most_common(limit):
        print(f"{word:<15} | {score:<12} | {count:<10}")

def run_diagnosis(df, sia):
    print("\n" + "="*60)
    print("PHASE 1: DIAGNOSIS (Auditing Original Model)")
    print("="*60)
    
    df['Numeric_Score'] = pd.to_numeric(df['Score'], errors='coerce')
    valid_df = df.dropna(subset=['Numeric_Score'])

    is_high_score = (
        ((valid_df['Type'] == 'Critic') & (valid_df['Numeric_Score'] >= 70)) | 
        ((valid_df['Type'] == 'User') & (valid_df['Numeric_Score'] >= 7))
    )
    is_low_score = (
        ((valid_df['Type'] == 'Critic') & (valid_df['Numeric_Score'] <= 50)) | 
        ((valid_df['Type'] == 'User') & (valid_df['Numeric_Score'] <= 5))
    )


    temp_scores = valid_df['Raw Text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    
    # False Negatives
    fn_indices = valid_df[is_high_score & (temp_scores <= -0.05)].index
    fn_texts = valid_df.loc[fn_indices, 'Raw Text']
    
    # False Positives
    fp_indices = valid_df[is_low_score & (temp_scores >= 0.05)].index
    fp_texts = valid_df.loc[fp_indices, 'Raw Text']

    print(f"Detected {len(fn_texts)} False Negatives (Good Game -> Bad Sentiment)")
    print(f"Detected {len(fp_texts)} False Positives (Bad Game -> Good Sentiment)")

    fn_culprits = get_culprit_words(fn_texts, sia, target_type="negative")
    fp_culprits = get_culprit_words(fp_texts, sia, target_type="positive")

    print_diagnostic_table(
        "[1] Top Words punishing High-Rated Games (False Negatives)\nLogic: Words with Negative scores appearing in Good Games.", 
        fn_culprits
    )
    
    print_diagnostic_table(
        "[2] Top Words boosting Low-Rated Games (False Positives)\nLogic: Words with Positive scores appearing in Bad Games.", 
        fp_culprits
    )
    
    print("\n>>> INSIGHT: Based on this diagnosis, we will now apply a Custom Gaming Lexicon.")


def get_gaming_lexicon():
    return {
        'horror': 0.5,      
        'hell': 0.0,        
        'dead': 0.0,        
        'grim': 0.0,        
        'sin': 0.0,        
        'terror': 0.5,      
        'thief': 0.0,       
        'combat': 0.0,      
        'war': 0.0,         
        'enemies': 0.0,
        'kill': 0.0,
        'killing': 0.0,
        'battle': 0.0,
        'fire': 0.0,
        'shoot': 0.0,
        'fight': 0.0,
        'attack': 0.0,       
        'difficult': 0.2, 
        'punishing': 0.1, 
        'play': 0.0,
        'playing': 0.0,
        'played': 0.0,
        'adventure': 0.0,
        'pretty': 1.0,      
        'interesting': 1.0,
        'chore': -2.5,
        'finicky': -2.0,
        'broken': -3.0,
        'boring': -3.0,
        'flaw': -2.0,
        'flaws': -2.0,
        'repetitive': -2.5,
        'generic': -2.0,
    }

def run_final_analysis(df, sia_tuned):
    print("\n" + "="*60)
    print("PHASE 2: EXECUTION (Running Tuned Model)")
    print("="*60)
    
    print("Calculating final sentiment scores with Custom Lexicon...")
    df['Sentiment Score'] = df['Raw Text'].apply(lambda x: sia_tuned.polarity_scores(str(x))['compound'])
    
    def get_category(score):
        if score >= 0.05: return "Positive"
        elif score <= -0.05: return "Negative"
        else: return "Neutral"
        
    df['Sentiment Category'] = df['Sentiment Score'].apply(get_category)
    return df

def load_aspect_keywords(csv_path):
    if not os.path.exists(csv_path): return {}
    try:
        df = pd.read_csv(csv_path)
        aspect_dict = {}
        for _, row in df.iterrows():
            dimension = row['Dimension'].strip()
            keywords_str = row['Keywords']
            if isinstance(keywords_str, str):
                keywords = [k.strip().lower() for k in keywords_str.split(';') if k.strip()]
                aspect_dict[dimension] = keywords
        return aspect_dict
    except: return {}

def get_aspect_scores(text, sia, aspect_keywords):
    if not isinstance(text, str): return {k: None for k in aspect_keywords.keys()}
    try: sentences = sent_tokenize(text)
    except LookupError: sentences = [text]
    
    aspect_scores_map = {k: [] for k in aspect_keywords.keys()}
    for sentence in sentences:
        sent_lower = sentence.lower()
        for aspect, keywords in aspect_keywords.items():
            if any(kw in sent_lower for kw in keywords):
                aspect_scores_map[aspect].append(sia.polarity_scores(sentence)['compound'])
    
    return {k: (sum(v)/len(v) if v else None) for k, v in aspect_scores_map.items()}


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    input_path = os.path.join(project_root, "data", "processed", "processed_reviews.csv")
    aspect_path = os.path.join(project_root, "data", "raw","attribution_dimensions_12.csv")
    output_path = os.path.join(project_root, "data", "processed", "analyzed_reviews.csv")

    if not os.path.exists(input_path):
        print("Error: Input file not found.")
        return

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    sia_base = SentimentIntensityAnalyzer()
    run_diagnosis(df, sia_base)

    sia_tuned = SentimentIntensityAnalyzer()
    custom_lexicon = get_gaming_lexicon()
    sia_tuned.lexicon.update(custom_lexicon)
    print(f"\n[Action] Updated VADER lexicon with {len(custom_lexicon)} custom weights based on diagnosis.")
    
    df = run_final_analysis(df, sia_tuned)
    
    print("\nRunning Aspect-Based Sentiment Analysis...")
    aspect_keywords = load_aspect_keywords(aspect_path)
    if aspect_keywords:
        aspect_data = df['Raw Text'].apply(lambda x: get_aspect_scores(str(x), sia_tuned, aspect_keywords))
        aspect_df = pd.json_normalize(aspect_data)
        aspect_df.columns = [f"Sentiment_{col}" for col in aspect_df.columns]
        df = pd.concat([df.reset_index(drop=True), aspect_df.reset_index(drop=True)], axis=1)
    
    df.to_csv(output_path, index=False)
    print(f"\nSuccess! Final analyzed data saved to: {output_path}")

if __name__ == "__main__":
    main()
