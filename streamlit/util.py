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

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_vertexai import ChatVertexAI


# from langchain_core.messages import HumanMessage
# from langchain_core.prompts.image import ImagePromptTemplate
# from langchain.prompts import PromptTemplate
# from langchain_core.prompt_values import ImageURL
# from langchain_core.pydantic_v1 import BaseModel, Field, validator
# from langchain.output_parsers import PydanticOutputParser



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
    meta_cols = [
        'ITEM_ID', 'SIMILARITY', 'NAME', 'CATEGORY', 'PRICE', 'SELLABLE_ONLINE', 
        'LINK', 'DESIGNER', 'SHORT_DESCRIPTION', 'DEPTH', 'HEIGHT', 'WIDTH'
    ]
    subset_cols = [
        'ITEM', 'ITEM_ID', 'NAME', 'CATEGORY', 'PRICE($)','SIMILARITY', 'DEPTH', 'HEIGHT', 'WIDTH', 'LINK'
    ]
    
    output = []
    for res in result:
        output.append(pd.DataFrame(list(res)).T)
    result_df = pd.concat(output).reset_index(drop=True)
    result_df.columns = meta_cols
    
    result_df["ITEM"] = result_df.ITEM_ID.apply(lambda x: f"{args.GCS_BUCKET}/{args.IMG_DIR}/{x}.jpg")
    result_df.rename(columns = {'PRICE' : 'PRICE($)'}, inplace = True)

    return result_df[subset_cols]

def path_to_image_html(path):
    if not path.startswith("https://"):
        return '<img src="https://storage.cloud.google.com/' + path + '?authuser=1" width="200" >'
    else:
        return '<a href="' + path + '">purchase_link</a>'

class SearchKeyword(BaseModel):
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
    # json_model = model_to_json(SearchKeyword(keyword='', reason=''))
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
        
        Now create a "keyword" and the "reason" for creating this keyword in a structured JSON format that matches the following model: 
        """
    ##{json_model}.
    return prompt

def get_masking_img(img, pos):
    left_top_x = int(pos[0])
    left_top_y = int(pos[1])
    right_bottom_x = int(pos[2])
    right_bottom_y = int(pos[3])

    height, width, dim = np.shape(img)

    height, width, dim = np.shape(img)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[left_top_y:right_bottom_y, left_top_x:right_bottom_x] = 255

    return mask


# class TagsForAspects(BaseModel):

#     room_sentiment: list[str] = Field(..., description = "The sentiments of the room");

#     room_color: list[str] = Field(..., description = "The overall atmospheres that depicted to a color");

#     room_size: list[int] = Field(..., description = "room size expressed in square meters", enum = list(range(1, 100)));

#     housing_type: list[str] = Field(..., description = "The type of housing",
#                               enum = ['studio', 'apartment', 'house']);

#     room_type: list[str] = Field(..., description = "The type of room",
#                            enum = ['living room', 'kitchen', 'home office', 'bedroom', 'bathroom', 'dining room', 'office', 'garage', 'basement', 'attic', 'laundry room', 'pantry', 'family room', 'foyer']);


# class SentenceToAspect:

#     def __init__(self):
#         GOOGLE_API_KEY = "AIzaSyCVFmbgbdktlHrE9_w3c9CeWo3Dchf6Of0"
#         self.GOOGLE_API_KEY = GOOGLE_API_KEY
#         self.llm = ChatGoogleGenerativeAI(
#             model = 'gemini-pro',
#             google_api_key = self.GOOGLE_API_KEY,
#             temperature = 0
#             )

#         self.parser = PydanticOutputParser(pydantic_object = TagsForAspects)
#         self.prompt = PromptTemplate(
#             template = """Answer the user query. \n {format_instructions}\n{query}\n

#             1. You must extract the formatted aspect from each word or keyword within query sentence, rather than the sentence as a whole.
#             2. Please concentrate metric information if it is considerated in the original human message.
#             3. If you don't think that there is no appropriate words for those aspects, you must return the word 'none' to the aspect
#             """,
#             input_variables = ["query"],
#             partial_variables = {"format_instructions" : self.parser.get_format_instructions()}
#         )

#     def query(self, query_sentence: str):

#         chain = self.prompt | self.llm | self.parser
#         return chain.invoke({"query" : query_sentence})

