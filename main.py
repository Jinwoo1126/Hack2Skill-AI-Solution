import os
import json
from dotenv import load_dotenv
from src.data_pipeline import get_shopping_data, get_thumbnail, get_ikea_data

# load .env
load_dotenv()

# get the api key
API_KEYS = {
    'SERP_API': os.environ.get('SERP_API')
}

with open('config.json', 'r') as f:
    config = json.load(f)

# get the keyword
for keyword in config['list_of_furnitures']:
    shopping_results = get_shopping_data(query=keyword, api_key=API_KEYS['SERP_API'])

    if not os.path.exists('data'):
        os.makedirs('data')

    for shopping_result in shopping_results:
        shopping_result = get_thumbnail(shopping_dict=shopping_result, keywords=keyword)

    # save the shopping results to a JSON file
    with open('data/'+ keyword +'_shopping_results.json', 'w') as f:
        json.dump(shopping_results, f, indent=4)

# ikea_dataset 
        
'''
ikea_meta = get_ikea_data('data/ikea')
with open('data/ikea/ikea_results.json', 'w') as f:
    json.dump(ikea_meta, f, indent=4)
'''
    
