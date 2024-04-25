import uvicorn
import os
from fastapi import Depends, FastAPI, HTTPException

from starlette import status
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from routers import inference
from model import diffusion_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_app() -> FastAPI:
    app = FastAPI(title="Inference API")
    diffusion_model.init_app(app)

    app.include_router(
        inference.router, prefix=f"/inference", tags=["Diffusion model inference"]
    )
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
