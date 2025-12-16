import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import os

print("Checking NLTK resources...")
resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']

for resource in resources:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            print(f"Downloading missing resource: {resource}...")
            nltk.download(resource, quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text_pipeline(text, game_title):
    if not isinstance(text, str):
        return []
    
    text = text.lower()
    
    # Process Game Title to create dynamic stopwords
    title_tokens = set()
    if isinstance(game_title, str):
        raw_title_tokens = word_tokenize(game_title.lower())
        for t in raw_title_tokens:
            if t not in string.punctuation and t.isalnum():
                title_tokens.add(t)

    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
        tokens = word_tokenize(text)
    
    clean_tokens = []
    for token in tokens:
        if token in string.punctuation or not token.isalpha():
            continue
            
        if token in stop_words:
            continue
        
        # Remove Game Title words
        if token in title_tokens:
            continue
            
        lemma = lemmatizer.lemmatize(token)
        
        if lemma not in stop_words and lemma not in title_tokens:
            clean_tokens.append(lemma)
        
    return clean_tokens

def flatten_reviews(games_data):
    all_reviews = []
    
    for game in games_data:
        game_title = game.get("Title", "Unknown")
        developer = game.get("Developer", "Unknown") 
        publisher = game.get("Publisher", "Unknown") 
        
        def extract_reviews(review_list, user_type):
            for r in review_list:
                if r.get("text"):
                    all_reviews.append({
                        "Game": game_title,
                        "Developer": developer,
                        "Publisher": publisher,
                        "Type": user_type,
                        "Score": r.get("score"),
                        "Raw Text": r.get("text")
                    })

        if "Critic Reviews Data" in game:
            extract_reviews(game["Critic Reviews Data"], "Critic")
                
        if "User Reviews Data" in game:
            extract_reviews(game["User Reviews Data"], "User")
                
    return all_reviews

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    input_path = os.path.join(project_root, "data", "raw", "games_dataset.json")
    output_path = os.path.join(project_root, "data", "processed", "processed_reviews.csv")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    print(f"Loading raw data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        games_data = json.load(f)
    
    print("Flattening review structure...")
    reviews_list = flatten_reviews(games_data)
    df = pd.DataFrame(reviews_list)
    print(f"Extracted {len(df)} total reviews.")

    print("Running NLP pipeline (Tokenization, Title Removal, Lemmatization)...")
    
    df['Processed Tokens'] = df.apply(
        lambda row: clean_text_pipeline(row['Raw Text'], row['Game']), 
        axis=1
    )
    
    df['Processed Text'] = df['Processed Tokens'].apply(lambda x: " ".join(x))

    df.to_csv(output_path, index=False)
    print(f"Success! Processed data saved to: {output_path}")

if __name__ == "__main__":
    main()