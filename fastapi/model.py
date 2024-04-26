import torch
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline
from fastapi import FastAPI, HTTPException


class Model:
    def __init__(self, device='cpu'):
        self.device = device
        self.pipe = None

    def init_app(self, app: FastAPI):
        @app.on_event("startup")
        def startup():
            _torch_dtype = torch.float32 if self.device == "cpu" else torch.float16
            self.pipe = DiffusionPipeline.from_pretrained(
                            "Fantasy-Studio/Paint-by-Example",
                            use_onnx=True,
                            torch_dtype=_torch_dtype,
                        )
            self.pipe = self.pipe.to(self.device)

            self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                                    "runwayml/stable-diffusion-v1-5",
                                    use_onnx=True,
                                    torch_dtype=_torch_dtype,
                                )
            self.img2img_pipe = self.img2img_pipe.to(self.device)

        @app.on_event("shutdown")
        def shutdown():
            self.pipe = None
            self.img2img_pipe = None

diffusion_model = Model()

