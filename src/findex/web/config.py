from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)
    index_path: Path = Path("index.bin")
    corpus_dir: Path = Path("data")
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

settings = Settings()
