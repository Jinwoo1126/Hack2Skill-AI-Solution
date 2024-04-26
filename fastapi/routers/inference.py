import os
import io
import base64
from typing import List, Dict, Optional

from pydantic import Field
from fastapi import (
    APIRouter,
    HTTPException,
    status,
)
from fastapi.responses import JSONResponse, Response

from PIL import Image
from model import diffusion_model
from schema import Input, InputImg2Img

router = APIRouter()
IMG_SIZE = (512, 512)

def image2bytes(image: Image) -> bytes:
    byte_array = io.BytesIO()
    _format = image.format if image.format is not None else 'PNG'
    image.save(byte_array, format=_format)
    byte_array = byte_array.getvalue()
    return byte_array


@router.get(
    "/check/alive",
    summary="Health Check",
    description="",
)
def get_health() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/",
    summary="Paint-by-Example model inference",
    description="",
)
def predict(inputs:Input) -> bytes:

    base_image = Image.open(io.BytesIO(base64.b64decode(inputs.base_bytes)))
    mask_image = Image.open(io.BytesIO(base64.b64decode(inputs.mask_bytes)))
    example_image = Image.open(io.BytesIO(base64.b64decode(inputs.example_bytes)))

    image = diffusion_model.pipe(
                image=base_image,
                mask_image=mask_image, 
                example_image=example_image,
                guidance_scale=15,
                num_inference_steps=50
    ).images[0]

    image_bytes = image2bytes(image)
    encoded_bytes = base64.b64encode(image_bytes)
    
    return encoded_bytes.decode('ascii')

@router.post(
    "/img2img",
    summary="Image-to-Image model inference",
    description="",
)
def predict_img2img(inputs:InputImg2Img) -> bytes:

    base_image = Image.open(io.BytesIO(base64.b64decode(inputs.base_bytes)))
    
    image = diffusion_model.img2img_pipe(
                prompt=inputs.prompt,
                negative_prompt=inputs.negative_prompt,
                image=base_image,
                guidance_scale=15,
                num_inference_steps=50
    ).images[0]

    image_bytes = image2bytes(image)
    encoded_bytes = base64.b64encode(image_bytes)
    
    return encoded_bytes.decode('ascii')

