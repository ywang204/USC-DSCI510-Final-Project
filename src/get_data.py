import requests
import json
import time
import pandas as pd
import os
import re
from bs4 import BeautifulSoup
import random
from typing import List, Dict, Optional, Any

API_KEY = "9c980291eemsh81ffc14c0dbab38p1ed876jsn2012d20207db"
API_HOST = "metacritic-api1.p.rapidapi.com"
API_URL = "https://metacritic-api1.p.rapidapi.com/game"

def get_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_raw = os.path.join(project_root, 'data', 'raw')
    os.makedirs(data_raw, exist_ok=True)
    return data_raw

def scrape_titles(start_page: int, end_page: int) -> List[str]:
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
                print(f"Failed to retrieve page {page}. Status code: {response.status_code}")               
        except Exception as e:
            print(f"Error on page {page}: {e}")           
        
        sleep_time = random.uniform(2.0, 4.0)
        time.sleep(sleep_time)        
    return titles
    
def generate_titles() -> None:
    all_target_games = []
    print("\n--- Phase 1: Collecting Top Tier Games ---")
    all_target_games.extend(scrape_titles(1, 4))
    
    print("\n--- Phase 2: Collecting Mid Tier Games ---")
    all_target_games.extend(scrape_titles(130, 132))
    
    print("\n--- Phase 3: Collecting Low Tier Games ---")
    all_target_games.extend(scrape_titles(261, 264))
    
    unique_games = list(set(all_target_games))
    print(f"\nTotal unique games collected: {len(unique_games)}")

    output_dir = get_paths()
    config_path = os.path.join(output_dir, "target_game_list.json")

    print(f"Saving game list to: {config_path}")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(unique_games, f, indent=4)
        
    print(f"\nGenerated Successfully")

def load_target_games() -> List[str]:
    output_dir = get_paths()
    target_list_path = os.path.join(output_dir, "target_game_list.json")
    
    if os.path.exists(target_list_path):
        try:
            with open(target_list_path, 'r', encoding='utf-8') as f:
                games = json.load(f)
            print(f"Loaded successfully, {len(games)} games ready for scraping.")
            return games
        except Exception as e:
            print(f"Error loading JSON list: {e}")
            return []
    else:
        print(f"Target list not found at {target_list_path}. Generating now...")
        generate_titles()
        if os.path.exists(target_list_path):
            with open(target_list_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

def fetch_raw_data(game_title: str) -> Optional[Dict[str, Any]]:
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
            print(f"Cannot find game: {slug}")
            return None
        else:
            print(f"Status: {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def clean_data(data: Dict[str, Any]) -> Dict[str, Any]:
    title = data.get("title", "Unknown")
    platform = data.get("platform", "Unknown")
    release_date = data.get("release_date", "Unknown")
    genre = data.get("genre", "Unknown")   
    developers_list = data.get("developer", [])
    developer = ", ".join(developers_list) if isinstance(developers_list, list) else str(developers_list)
    publishers_list = data.get("publisher", [])
    publisher = ", ".join(publishers_list) if isinstance(publishers_list, list) else str(publishers_list)

    if "critic_reviews" in data and isinstance(data["critic_reviews"], dict):
        metascore = data["critic_reviews"].get("metascore")
        critic_review_count = data["critic_reviews"].get("review_count")
    else:
        metascore = None
        critic_review_count = 0
    
    if "user_reviews" in data and isinstance(data["user_reviews"], dict):
        user_score = data["user_reviews"].get("user_score")
        user_review_count = data["user_reviews"].get("review_count")
    else:
        user_score = None
        user_review_count = 0
    
    critic_details = []
    if "critic_reviews" in data and isinstance(data["critic_reviews"], dict):
        latest = data["critic_reviews"].get("latest_reviews", [])
        if isinstance(latest, list):
            for review in latest:
                if review.get("review_text"):
                    critic_details.append({
                        "score": review.get("rating"),
                        "text": review.get("review_text"),
                    })

    user_details = []
    if "user_reviews" in data and isinstance(data["user_reviews"], dict):
        latest = data["user_reviews"].get("latest_reviews", [])
        if isinstance(latest, list):
            for review in latest:
                if review.get("review_text"):
                    user_details.append({
                        "score": review.get("rating"),
                        "text": review.get("review_text"),
                    })

    return {
        "Title": title,
        "Platform": platform,
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
    output_dir = get_paths()
    print(f"Starting fetch process using output directory: {output_dir}")
    
    target_games_list = load_target_games()
    
    if not target_games_list:
        print("No target games found. Exiting.")
        return
    
    all_games_data = []
    
    for game in target_games_list:
        raw_json = fetch_raw_data(game)       
        if raw_json:
            if isinstance(raw_json, list) and len(raw_json) > 0:
                target_data = raw_json[0] 
            elif isinstance(raw_json, dict):
                target_data = raw_json
            else:
                print(f"No data for '{game}'")
                continue                
            try:
                cleaned = clean_data(target_data)
                all_games_data.append(cleaned)
                print(f"Successfully fetched: {cleaned.get('Title')}")
            except Exception as parse_error:
                print(f"Error parsing {game}: {parse_error}")
        
        time.sleep(1.5)

    if all_games_data:
        df = pd.DataFrame(all_games_data)
        csv_path = os.path.join(output_dir, "games_dataset.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nData saved in csv: {csv_path}")
        
        json_path = os.path.join(output_dir, "games_dataset.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_games_data, f, ensure_ascii=False, indent=4)
        print(f"Data saved in json: {json_path}")
    else:
        print("\nNo data fetched")

if __name__ == "__main__":
    main()