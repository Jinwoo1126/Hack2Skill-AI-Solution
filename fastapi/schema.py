from pydantic import BaseModel

class Input(BaseModel):
    base_bytes: bytes
    mask_bytes: bytes
    example_bytes: bytes

class InputImg2Img(BaseModel):
    prompt: str
    negative_prompt: str
    base_bytes: bytes
