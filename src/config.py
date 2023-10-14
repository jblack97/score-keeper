import os
import json
from pydantic_settings import BaseSettings
from pathlib import Path

ROOT = Path(__file__).parent

with open(ROOT / "credentials.json", "r") as f:
    openai_api_key = json.loads(f.read())["openai_api_key"]


class BaseConfig(BaseSettings):
    openai_api_key: str = openai_api_key
    gpt_model_name: str = "gpt-3.5-turbo"


config = BaseConfig()
