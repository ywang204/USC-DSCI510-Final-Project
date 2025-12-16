import requests
import json
import time
import pandas as pd
import os
import re
from bs4 import BeautifulSoup
import random
from utils.paths import DATA_RAW, TARGET_GAME_LIST, ensure_dir

def scrape_titles(start_page, end_page):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.google.com/'
    }
    titles = []
    
    for page in range(start_page, end_page + 1):
        url = f"https://www.metacritic.com/browse/game/pc/all/all-time/metascore/?page={page}"
        print(f"Scraping page {page} ...")
        
        try:
            response = requests.get(url, headers=headers, timeout=10)           
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                cards = soup.find_all('div', class_='c-finderProductCard')         
                if not cards:
                    print(f"Warning: No game cards found on page {page}. Structure might have changed.")               
                for card in cards:
                    title_tag = card.find('h3', class_='c-finderProductCard_titleHeading')
                    if title_tag:
                        raw_text = title_tag.get_text(strip=True)
                        clean_title = raw_text.split('.', 1)[-1].strip()                      
                        titles.append(clean_title)
            else:
                print(f"  Failed to retrieve page {page}. Status code: {response.status_code}")               
        except Exception as e:
            print(f"Error on page {page}: {e}")           
        sleep_time = random.uniform(2.0, 4.0)
        time.sleep(sleep_time)        
    return titles
    
def generate_titles():
    all_target_games = []
    print("\n--- Phase 1: Collecting Top Tier Games ---")
    all_target_games.extend(scrape_titles(1, 4))
    
    print("\n--- Phase 2: Collecting Mid Tier Games ---")
    all_target_games.extend(scrape_titles(130, 132))
    
    print("\n--- Phase 3: Collecting Low Tier Games ---")
    all_target_games.extend(scrape_titles(261, 264))
    
    unique_games = list(set(all_target_games))
    print(f"\nTotal unique games collected: {len(unique_games)}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    output_dir = os.path.join(project_root, 'data', 'raw')
    
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, "game_config.py")

    print(f"Saving game list to: {config_path}")

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"TARGET_GAMES = {json.dumps(unique_games, indent=4)}")
        
    print(f"\nGenarated Successfully")
    
generate_titles()
    
API_KEY = "9c980291eemsh81ffc14c0dbab38p1ed876jsn2012d20207db"
API_HOST = "metacritic-api1.p.rapidapi.com"
API_URL = "https://metacritic-api1.p.rapidapi.com/game"

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
output_dir = os.path.join(project_root, 'data', 'raw')
os.makedirs(output_dir, exist_ok=True)

target_list_path = os.path.join(output_dir, "target_game_list.json")
TARGET_GAMES = []
if os.path.exists(target_list_path):
    try:
        with open(target_list_path, 'r', encoding='utf-8') as f:
            TARGET_GAMES = json.load(f)
        print(f"Loaded successfully, {len(TARGET_GAMES)} games ready for scraping from {target_list_path}")
    except Exception as e:
        print(f"Error loading JSON list: {e}")
else:
    try:
        from game_config import TARGET_GAMES
        print(f"Loaded from config file, {len(TARGET_GAMES)} games ready.")
    except ImportError:
        print(f"Error: Could not find target_game_list.json at {target_list_path} nor game_config.py")
        TARGET_GAMES = []

def fetch_raw_data(game_title):
    slug = game_title.lower()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = slug.strip().replace(' ', '-')
    target_url = f"{API_URL}/{slug}"   
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": API_HOST
    }  
    try:
        response = requests.get(target_url, headers=headers)        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            print(f" Can find game: {slug}")
            return None
        else:
            print(f"Status: {response.status_code}): {response.text}")
            return None
    except Exception as e:
        print(f"Error{e}")
        return None

def clean_data(data):
    title = data.get("title", "Unknown")
    platform = data.get("platform", "Unknown")
    release_date = data.get("release_date", "Unknown")
    genre = data.get("genre", "Unknown")   
    developers_list = data.get("developer", [])
    developer = ", ".join(developers_list)
    publishers_list = data.get("publisher", [])
    publisher = ", ".join(publishers_list)

    if "critic_reviews" in data:
        metascore = data["critic_reviews"].get("metascore")
        critic_review_count = data["critic_reviews"].get("review_count")
    else:
        metascore = None
        critic_review_count = 0
    
    if "user_reviews" in data:
        user_score = data["user_reviews"].get("user_score")
        user_review_count = data["user_reviews"].get("review_count")
    else:
        user_score = None
        user_review_count = 0
    
    critic_details = []
    if "critic_reviews" in data and "latest_reviews" in data["critic_reviews"]:
        for review in data["critic_reviews"]["latest_reviews"]:
            if review.get("review_text"):
                critic_details.append({
                    "score": review.get("rating"),  # Usually 0-100
                    "text": review.get("review_text"),
                })

    user_details = []
    if "user_reviews" in data and "latest_reviews" in data["user_reviews"]:
        for review in data["user_reviews"]["latest_reviews"]:
            if review.get("review_text"):
                user_details.append({
                    "score": review.get("rating"), # Usually 0-10
                    "text": review.get("review_text"),
                })

    return {
        "Title": title,
        "Platform":platform,
        "Release Date": release_date,
        "Genre": genre,
        "Developer": developer,
        "Publisher": publisher,
        "Metascore": metascore,
        "Critic Review Count": critic_review_count,
        "User Score": user_score,
        "User Review Count": user_review_count,
        "Critic Reviews Data": critic_details, 
        "User Reviews Data": user_details
    }

def main():
    all_games_data = []
    
    print(f"Starting fetch process using output directory: {output_dir}")
    
    if not TARGET_GAMES:
        print("No target games found. Exiting.")
        return
    
    for game in TARGET_GAMES:
        raw_json = fetch_raw_data(game)       
        if raw_json:
            if isinstance(raw_json, list) and len(raw_json) > 0:
                target_data = raw_json[0] 
            elif isinstance(raw_json, dict):
                target_data = raw_json
            else:
                print(f" No data of '{game}' ")
                continue                
            try:
                cleaned_data = clean_data(target_data)
                all_games_data.append(cleaned_data)
                print(f"Successfully fetch:{cleaned_data.get('Title')}")
            except Exception as parse_error:
                print(f"Error:{parse_error}")
        time.sleep(2)

    if all_games_data:
        df = pd.DataFrame(all_games_data)
        csv_path = os.path.join(output_dir, "games_dataset.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n Data in csv:{csv_path}")
        
        json_path = os.path.join(output_dir, "games_dataset.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_games_data, f, ensure_ascii=False, indent=4)
        print(f"\n Data in json: {json_path}")
        
    else:
        print("\n No data fetched")

if __name__ == "__main__":
    main()