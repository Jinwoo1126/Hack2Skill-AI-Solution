
import os
from serpapi import search
from urllib import request


def get_shopping_data(query:str=None, api_key:str=None, **kwargs) -> dict:
    """
    Retrieves shopping data based on the provided query and API key.

    Args:
        query (str, optional): The search query. Defaults to None.
        api_key (str, optional): The API key for accessing the shopping data. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        dict: A dictionary containing the shopping results.
    """
    params = {
    "engine": "google_shopping",
    "q": query,
    "location": "United States",
    "hl": "en",
    "gl": "us",
    "api_key": api_key
    }

    search_result = search(params)
    shopping_results = search_result['shopping_results']

    return shopping_results


def get_thumbnail(shopping_dict: dict = None, keywords: str = None) -> dict:
    """
    Downloads and saves the thumbnail image from the given URL.

    Args:
        shopping_dict (dict): A dictionary containing information about the shopping item.
        keywords (str): The keywords associated with the shopping item.

    Returns:
        dict: The updated shopping_dict with the thumbnail image path.

    """
    url = shopping_dict['thumbnail']
    savepath = os.path.join('data', keywords, 'thumbnails')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savename = os.path.join(savepath, str(shopping_dict['position']) + '.jpg')
    shopping_dict['thumbnail'] = savename

    try:
        request.urlretrieve(url, savename)
    except Exception as e:
        print(e)

    return shopping_dict

