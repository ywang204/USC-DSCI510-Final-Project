import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_RAW = os.path.join(ROOT_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(ROOT_DIR, 'data', 'processed')

RAW_GAMES_JSON = os.path.join(DATA_RAW, 'games_dataset.json')
PROCESSED_REVIEWS_CSV = os.path.join(DATA_PROCESSED, 'processed_reviews.csv')
TARGET_GAME_LIST = os.path.join(DATA_RAW, 'target_game_list.json')

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)