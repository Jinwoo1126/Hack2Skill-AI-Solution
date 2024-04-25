import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_ID: str
    PROJECT_NUM: str
    KEY_PATH: str
    REGION: str
    SCOPES:list = ["https://www.googleapis.com/auth/cloud-platform"]
    
    # gcs
    GCS_BUCKET: str
    IMG_DIR: str
    
    # vertex ai
    API_ENDPOINT: str
    INDEX_ENDPOINT_ID: str
    DEPLOYED_INDEX_ID: str
    
    # alloy
    URI: str
    PRIVATE_IP: str
    DB_REGION: str
    CLUSTER_ID: str
    INSTANCE_ID: str
    DB_NAME: str
    DB_USER: str
    DB_PW: str

    
    
    class Config:
        env_file = ".env"


args = Settings(_env_file=f'.env')