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


def get_ikea_data(data_path):
    df = pd.read_csv(os.path.join(data_path, 'IKEA.csv'))
    data_save_path = "{}/images".format(data_path)
    meta_dict = {}
    make_dirs(data_save_path)
            
    for idx in tqdm(range(df.shape[0])):
        item_id = df.loc[idx,'item_id']
        name = df.loc[idx,'name']
        link = df.loc[idx,'link']
        try:
            page = requests.get(link)
            soup = bs(page.text, "html.parser")
            image_tag =  soup.find('div', {"class": "pip-product-gallery__media pip-product-gallery__media--active"}).find('img')
            image_url = image_tag.get('src')
            response = requests.get(image_url)
        
            with open('{}/{}_{}.jpg'.format(data_save_path, item_id, name), 'wb') as f:
                f.write(response.content)
            
            meta_dict["{}_{}".format(item_id, name)] = {
                "link" : link,
                "category" : df.loc[idx, 'category'],
                "price" : df.loc[idx, 'price'],
                "old_price" : df.loc[idx, 'old_price'],
                "designer" : df.loc[idx, 'designer'],
                "depth" : str(df.loc[idx, ['depth']].fillna(0).values[0]),
                "height" : str(df.loc[idx, ['height']].fillna(0).values[0]),
                "width" : str(df.loc[idx, ['width']].fillna(0).values[0])
            }
        except:
            continue
        
    return meta_dict
