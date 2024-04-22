import json
from google.oauth2 import service_account
from google.cloud import storage, aiplatform_v1, vision

from indexing import EmbeddingResponse, EmbeddingPredictionClient
from util import *
from variable import args
from client import Client

import streamlit as st

HALF_IMG_SIZE = 650
REC_IMG_SIZE = 200

client = Client(args)

# @st.cache_resource
def convert_df(input_df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    input_df.style.set_table_styles(
       [{
           'selector': 'th',
           'props': [
               ('background-color', 'lightgrey'),
               #('color', 'cyan')
           ]
       }]
    )
    return input_df.to_html(escape=False, index=False, justify='center', formatters=dict(ITEM=path_to_image_html))

# streamlit page options
st.set_page_config(
     layout="wide",
     page_title=args.PROJECT_ID,
)

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown("setting arguments")
max_item = st.sidebar.slider('Max Items', 5, 10, 5, 5)
threshold = st.sidebar.slider('Distance Threshold', 0.0, 1.0, 0.5, 0.05)
search_type = st.sidebar.radio(label = 'Search Type', options = ['Image', 'Text'])
st.sidebar.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.title("[Hack2Skill] AI Solution Team")
st.subheader("Recommender System using Generative AI")

uploaded_file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])
prompt = st.text_input('Text what you want: ')


objects = []
if uploaded_file is not None and prompt is not None:
    left, right = st.columns(2)
    img, img_ary = load_uploaded_image(uploaded_file)
    org_img = img.copy()
    
    left.image(img_ary, width=HALF_IMG_SIZE)

    # object detection
    detected_img, objects = client.draw_bbox(img)
    right.image(detected_img, width=HALF_IMG_SIZE)


if len(objects):
    target_obj = st.selectbox(
        "Choose an item", options=objects
    )
    cropped_img = org_img.crop(objects[target_obj])
    cropped_img.format = img.format
    st.image(cropped_img, width=REC_IMG_SIZE)

    run_yn = st.button("Search")
    if run_yn:     
        results, response, generated_text = client.get_recommended_items(
            text=prompt, 
            image_bytes=image2bytes(cropped_img),
            threshold=threshold, 
            max_item=max_item,
            search_type=search_type,
        )
        
        html = convert_df(response)
        
        st.write("")
        if len(generated_text):
            st.write(f"search keword: {generated_text['keyword']}")
            st.write(f"reason: {generated_text['reason']}")
        st.write("Recommended Items :")
        if len(results):
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.write("Sorry! ")

        # if len(results):
        #     st.write("")
        #     st.image(results, width=200)
        # else:
        #     st.write("죄송합니다! 검색 쿼리와 일치하는 이미지가 없습니다. 다시 시도해 주세요.")