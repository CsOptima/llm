from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_ID: str = "ruslandev/llama-3-8b-gpt-4o-ru1.0"
    device: str = "mps"

    class Config:
        env_file = ".env"


settings = Settings()
