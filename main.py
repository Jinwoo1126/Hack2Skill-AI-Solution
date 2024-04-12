import os
import json
from dotenv import load_dotenv
from src.data_pipeline import get_shopping_data, get_thumbnail

# load .env
load_dotenv()

# get the api key
API_KEYS = {
    'SERP_API': os.environ.get('SERP_API')
}

keywords = 'chair'

shopping_results = get_shopping_data(query=keywords, api_key=API_KEYS['SERP_API'])

if not os.path.exists('data'):
    os.makedirs('data')

for shopping_result in shopping_results:
    shopping_result = get_thumbnail(shopping_dict=shopping_result, keywords=keywords)

# save the shopping results to a JSON file
with open('data/shopping_results.json', 'w') as f:
    json.dump(shopping_results, f, indent=4)

    
