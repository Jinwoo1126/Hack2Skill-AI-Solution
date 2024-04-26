import json
import io
from google.oauth2 import service_account
from google.cloud import storage, aiplatform_v1, vision

from indexing import EmbeddingResponse, EmbeddingPredictionClient
from util import *
from diffusion_utils import *
from variable import args
from client import Client

import streamlit as st
from streamlit_image_select import image_select

from PIL import Image



HALF_IMG_SIZE = 650
REC_IMG_SIZE = 200

client = Client(args)

def convert_df(input_df):
    return input_df.to_html(escape=False, index=False, justify='center', formatters=dict(ITEM=path_to_image_html, LINK=path_to_image_html))

# streamlit page options
st.set_page_config(
     layout="wide",
     page_title=args.PROJECT_ID,
)

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown("setting arguments")
max_item = st.sidebar.slider('Max Items', 5, 10, 5, 5)
search_type = st.sidebar.radio(label = 'Search Type', options = ['Image', 'Text'])
default_thr = 0.5 if search_type == 'Image' else 0.1
threshold = st.sidebar.slider('Similarity Threshold', 0.0, 1.0, default_thr, 0.05)
st.sidebar.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.title("[Hack2Skill] AI Solution Team")
st.subheader("Recommender System using Generative AI")

uploaded_file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])

objects = []

if uploaded_file is not None:
    if 'image' not in st.session_state:
        image = Image.open(uploaded_file)
        w, h = image.size
        print(f"loaded input image of size ({w}, {h})")
        width, height = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32
        image = image.resize((width, height))

        st.session_state.image = image

    st.write("Original Image")
    st.image(st.session_state.image, caption="Original Image", use_column_width=True)

    st.markdown("### Interior Guide")

    style_text = st.text_input("ex) change room to MID-Century style")
    prompt = style_text +  ", have to keep shape of the room, add furnitures in the room, high quality"
    st.markdown(f"**Prompt** : `{prompt}`")

    if st.button("Reset", type="primary"):
        st.experimental_rerun()

    if st.button("Generation"):
        negative_prompt = "change the shape of the room, do not change sytle, do not add furnitures, keep furnitures in the room, low quality"

        buffer = io.BytesIO()
        st.session_state.image.save(buffer, 'PNG')
        image_bytes = buffer.getvalue()

        img = IMG2IMG_API(image_bytes, prompt, negative_prompt)

        st.session_state.selected_img = img
            
    
    if st.button('Style Change'):
        negative_prompt = "change the shape of the room, do not change sytle, do not add furnitures, keep furnitures in the room, low quality"

        buffer = io.BytesIO()
        st.session_state.image.save(buffer, 'PNG')
        image_bytes = buffer.getvalue()

        imgs = client.edit_image_mask_free(image_bytes, prompt, negative_prompt)

        img_list = []
        for res in range(len(imgs.images)):
            img_list.append(Image.open(io.BytesIO(imgs.images[res]._image_bytes)))

        st.session_state.img_list = img_list

if 'img_list' in st.session_state:
    st.session_state.selected_img = image_select("Select Image", st.session_state.img_list)
    
if 'image' in st.session_state and prompt is not None:
    left, right = st.columns(2)

    if 'selected_img' in st.session_state:
        img = st.session_state.selected_img.copy()
        buffer = io.BytesIO()
        st.session_state.selected_img.save(buffer, 'PNG')
        buffer.seek(0)
        png_image = Image.open(buffer)
        org_img = png_image.copy()

        left.image(png_image, width=HALF_IMG_SIZE)
    
        # object detection
        detected_img, objects = client.draw_bbox(png_image)
        right.image(detected_img, width=HALF_IMG_SIZE)
    else:
        img, img_ary = load_uploaded_image(uploaded_file)
        org_img = img.copy()
        
        left.image(img, width=HALF_IMG_SIZE)
    
        # object detection
        detected_img, objects = client.draw_bbox(img)
        right.image(detected_img, width=HALF_IMG_SIZE)


if len(objects):
    target_obj = st.selectbox(
        "Choose an item", options=objects
    )
    st.session_state.cropped_img = org_img.crop(objects[target_obj])
    st.session_state.cropped_img.format = 'png'
    st.image(st.session_state.cropped_img, width=REC_IMG_SIZE)

    run_yn = st.button("Search")
    if run_yn:
        search_prompt = style_text + f'and target item is {target_obj}'
        st.session_state.recommend_results, st.session_state.response, st.session_state.generated_text = client.get_recommended_items(
            text=search_prompt,
            image_bytes=image2bytes(st.session_state.cropped_img),
            threshold=threshold, 
            max_item=max_item,
            search_type=search_type,
        )
        
        st.session_state.html = convert_df(st.session_state.response)

    if 'html' in st.session_state:
        with st.expander("Recommendation Results"):
            st.write("")
            if len(st.session_state.generated_text):
                st.markdown(f"`search keword`: {st.session_state.generated_text['keyword']}")
                st.markdown(f"`reason`: {st.session_state.generated_text['reason']}")
            st.markdown("**Recommended Items**")
            if len(st.session_state.recommend_results):
                st.markdown(st.session_state.html, unsafe_allow_html=True)
            else:
                st.write("Sorry! ")

    if 'recommend_results' in st.session_state:
        inpaint_item = st.selectbox(
            "Choose an item to inpainting", options=st.session_state.response.ITEM_ID.unique().tolist()
        )
        st.session_state.selected_item_img = st.session_state.recommend_results[inpaint_item]
        st.session_state.selected_item_img = Image.open(io.BytesIO(st.session_state.selected_item_img))
        st.image(st.session_state.selected_item_img)

    if 'selected_item_img' in st.session_state:
        inpaint_button = st.button("Inpaint")
        if inpaint_button:
            if 'selected_img' in st.session_state:
                st.session_state.mask_arr = get_masking_img(st.session_state.selected_img.copy(), objects[target_obj])
            elif 'image' in st.session_state:
                st.session_state.mask_arr = get_masking_img(st.session_state.image.copy(), objects[target_obj])
                    
            if st.session_state.mask_arr.sum() > 0:
                st.session_state.mask_img = Image.fromarray(st.session_state.mask_arr)

                with st.spinner('Wait for it...'):
                    st.session_state.inpainted_img = Paint_by_Example_API(st.session_state.selected_img, 
                                                                          st.session_state.mask_img, 
                                                                          st.session_state.selected_item_img)
            
                st.write("Inpainted")
                st.image(st.session_state.inpainted_img, output_format='PNG')
            elif 'mask' in st.session_state and 'image' in st.session_state:
                st.session_state.inpainted_img = Paint_by_Example_API(st.session_state.image,
                                                                      st.session_state.mask_img,
                                                                      st.session_state.selected_item_img)
            
                st.write("Inpainted")
                st.image(st.session_state.inpainted_img, output_format='PNG')
            else:
                st.write("Not Inpainted")
