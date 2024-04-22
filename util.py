import io
import re
import json
from typing import Tuple
from pydantic import BaseModel

import numpy as np
import pandas as pd

import pg8000
import sqlalchemy
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DatabaseError

from google.cloud import vision
from google.oauth2 import service_account
from google.cloud.alloydb.connector import Connector, IPTypes

from PIL import Image, ImageDraw

def load_uploaded_image(uploaded_image) -> (Image, np.array):
    img = Image.open(uploaded_image)
    return img, np.array(img)

def image2bytes(image: Image) -> bytes:
    byte_array = io.BytesIO()
    image.save(byte_array, format=image.format) #, format=image.format
    byte_array = byte_array.getvalue()
    return byte_array

def create_sqlalchemy_engine(
    args,
    credentials: service_account.Credentials
) -> Tuple[sqlalchemy.engine.Engine, Connector]:
    
    connector = Connector(credentials=credentials)

    def getconn() -> pg8000.dbapi.Connection:
        conn: pg8000.dbapi.Connection = connector.connect(
            instance_uri=args.URI,
            driver="pg8000",
            user=args.DB_USER,
            password=args.DB_PW,
            db=args.DB_NAME,
            ip_type="PRIVATE",
        )
        return conn

    # create SQLAlchemy connection pool
    engine = sqlalchemy.create_engine(
        "postgresql+pg8000://", creator=getconn, isolation_level="AUTOCOMMIT"
    )
    engine.dialect.description_encoding = None
    return engine, connector


def get_sql():
    query = sqlalchemy.text(
        f"""
        WITH SIM AS (
            SELECT item_id, 1-(embeddings <=> :value) as similarity
            FROM ikea_embeddings
            ORDER BY embeddings <-> :value
            LIMIT :max_item
        ) 
        SELECT A.item_id
             , A.similarity
             , B.name
             , B.category
             , B.price
             , B.sellable_online
             , B.link
             , B.designer
             , B.short_description
             , B.depth
             , B.height
             , B.width
          FROM SIM A
          LEFT JOIN (
              SELECT *
                FROM (
                  SELECT item_id
                       , name
                       , category
                       , price
                       , sellable_online
                       , link
                       , designer
                       , short_description
                       , depth
                       , height
                       , width
                       , ROW_NUMBER() OVER( PARTITION BY item_id ORDER BY category ) AS ROW_NUM
                  FROM ikea_items
              ) K
             WHERE K.ROW_NUM = 1
            ) B
        ON A.item_id = B.item_id
        ORDER BY A.similarity DESC
        ;
        """
    )
    return query

def get_result_table(args, result):
    cols = [
        'ITEM_ID', 'SIMILARITY', 'NAME', 'CATEGORY', 'PRICE', 'SELLABLE_ONLINE', 
        'LINK', 'DESIGNER', 'SHORT_DESCRIPTION', 'DEPTH', 'HEIGHT', 'WIDTH'
    ]
    output = []
    for res in result:
        output.append(pd.DataFrame(list(res)).T)
    result_df = pd.concat(output).reset_index(drop=True)
    result_df.columns = cols
    
    result_df["ITEM"] = result_df.ITEM_ID.apply(lambda x: f"{args.GCS_BUCKET}/{args.IMG_DIR}/{x}.jpg")

    return result_df[['ITEM', 'ITEM_ID', 'NAME', 'CATEGORY', 'PRICE', 'LINK', 'SIMILARITY']]

def path_to_image_html(path):
    if not path.startswith("https://"):
        return '<img src="https://storage.cloud.google.com/' + path + '?authuser=1" width="200" >'
    else:
        return '<img src="' + path + '" width="200" >'

class SearchKeword(BaseModel):
    keyword: str
    reason: str

def model_to_json(model_instance):
    return model_instance.model_dump_json()

def extract_json(text_response):
    # This pattern matches a string that starts with '{' and ends with '}'
    pattern = r'\{[^{}]*\}'
    matches = re.finditer(pattern, text_response)
    json_objects = []
    for match in matches:
        json_str = match.group(0)
        try:
            # Validate if the extracted string is valid JSON
            json_obj = json.loads(json_str)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            # Extend the search for nested structures
            extended_json_str = extend_search(text_response, match.span())
            try:
                json_obj = json.loads(extended_json_str)
                json_objects.append(json_obj)
            except json.JSONDecodeError:
                # Handle cases where the extraction is not valid JSON
                continue
    if json_objects:
        return json_objects
    else:
        return None  # Or handle this case as you prefer

def extend_search(text, span):
    # Extend the search to try to capture nested structures
    start, end = span
    nest_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            nest_count += 1
        elif text[i] == '}':
            nest_count -= 1
            if nest_count == 0:
                return text[start:i+1]
    return text[start:end]

def get_prompt(predefined_prompt:str) -> str:
    json_model = model_to_json(SearchKeword(keyword='', reason=''))
    prompt = \
        f"""
        You are the helpful assistant who generates appropriate shopping search keyword for the given image and user requirement.
        The user gave the following requirement for decorating room or living room, and there is an image of the furniture needed for this.
        **user requirement: {predefined_prompt}**
        
        Create the optimal shopping search keword according to the following steps.
        1.Extract key features of interior items in a given image.(e.g. color, material, size)
        2.Select the features that match the above features and **user requirement**.
        3.If there are no matching features, keep the main features in the image.
        4.Create the optimal shopping search keyword using the main characteristics of the item identified above.
        At this time, include 2 or 3 intuitive information such as color if possible and the keyword starts with the type of item.
        
        Now create a "keyword" and the "reason" for creating this keyword in a structured JSON format that matches the following model: {json_model}.
        """
    return prompt