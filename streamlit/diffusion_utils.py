# !pip install diffusers transformers
from PIL import Image
from io import BytesIO
import io
import os
import json
import base64

import requests

docker_net = os.environ.get('DOCKER_NET', '172.17.0.5:8000')
device = 'cpu'

def image2string(image):
    _bytes = image2bytes(image)
    encoded_bytes = base64.b64encode(_bytes)
    return encoded_bytes.decode('ascii')


def image2bytes(image: Image) -> bytes:
    byte_array = BytesIO()
    image.save(byte_array, format='PNG')
    byte_array = byte_array.getvalue()
    return byte_array
    

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")



def Paint_by_Example_API(base_image, mask_image, example_image):
    inference_url = f"http://{docker_net}/inference"

    print("api_call start")
    IMG_SIZE = (512, 512)
    
    init_image = base_image.resize(IMG_SIZE)
    mask_image = mask_image.resize(IMG_SIZE)
    example_image = example_image.resize(IMG_SIZE)

    reqest_body = {}
    
    reqest_body['base_bytes'] = image2string(init_image)
    reqest_body['mask_bytes'] = image2string(mask_image)
    reqest_body['example_bytes'] = image2string(example_image)

    res = requests.post(inference_url, data=json.dumps(reqest_body))

    decoded = base64.b64decode(res.content)
    converted_img = Image.open(io.BytesIO(decoded))
    print("api_call end")

    return converted_img


def IMG2IMG_API(base_image:bytes, prompt:str, negative_prompt:str=''):
    inference_url = f"http://{docker_net}/inference/img2img"

    print("api_call start")
    IMG_SIZE = (512, 512)
    
    encoded_init_bytes = base64.b64encode(base_image)
    
    data = {} 
    data['base_bytes'] = encoded_init_bytes.decode('ascii')
    data['prompt'] = prompt
    data['negative_prompt'] = negative_prompt

    res = requests.post(inference_url, data=json.dumps(data))

    decoded = base64.b64decode(res.content)
    converted_img = Image.open(io.BytesIO(decoded))
    print("api_call end")

    return converted_img
