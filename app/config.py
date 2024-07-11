from typing import Dict

from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

from app.pydantic_schemas.claims_extraction_task import ClaimsExtractionTask

load_dotenv(override=True)


class Settings(BaseSettings):
    PROJECT_NAME: str = os.getenv('PROJECT_NAME')
    OUR_SYSTEM_SECRET_KEY_TOKEN: str = os.getenv('OUR_SYSTEM_SECRET_KEY_TOKEN')
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY')
    AZURE_OPENAI_API_KEY: str = os.getenv('AZURE_OPENAI_API_KEY')
    AZURE_OPENAI_ENDPOINT: str = os.getenv('AZURE_OPENAI_ENDPOINT')
    AZURE_OPENAI_API_VERSION: str = os.getenv('AZURE_OPENAI_API_VERSION')
    AZURE_OPENAI_MODEL: str = os.getenv('AZURE_OPENAI_MODEL')
    AZURE_RESOURCE_NAME: str = os.getenv('AZURE_RESOURCE_NAME')
    ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY')
    CLAIMS_CHECKER_RESPONSES_TRACKER: Dict[str, ClaimsExtractionTask] = {}


settings = Settings()
