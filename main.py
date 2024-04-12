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

keyword = 'chair', 'bed', 

shopping_results = get_shopping_data(query=keyword, api_key=API_KEYS['SERP_API'])

if not os.path.exists('data'):
    os.makedirs('data')

for shopping_result in shopping_results:
    shopping_result = get_thumbnail(shopping_dict=shopping_result, keywords=keyword)

# save the shopping results to a JSON file
with open('data/'+ keyword +'_shopping_results.json', 'w') as f:
    json.dump(shopping_results, f, indent=4)

    
