from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    # MongoDB
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "kira"
    MONGO_COLL: str = "posts"
    ENABLE_CHANGE_STREAMS: bool = False

    # Models
    EMBED_MODEL_PATH: str = "./models/embed"
    RERANKER_PATH: str = "./models/reranker" 

    # Index
    INDEX_DIR: str = "./indexes"

    # Search defaults
    RETRIEVE_K: int = 200
    FINAL_TOPK: int = 10

    # Logging
    LOG_LEVEL: str = "INFO"

    # Performance
    EMBEDDING_BATCH_SIZE: int = 32
    MAX_QUERY_LENGTH: int = 1000
    MAX_CANDIDATES_FILTER: int = 50000

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

def load_settings() -> Settings:
    return Settings()  # .env est lu automatiquement