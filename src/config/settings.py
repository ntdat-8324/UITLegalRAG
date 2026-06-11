import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # LLM Settings
    openrouter_api_key: str = Field(default="")
    openrouter_model: str = Field(default="google/gemma-2-9b-it:free")
    
    # Paths
    chroma_persist_dir: str = Field(default="./chroma_db")
    data_dir: str = Field(default="./data/raw")
    processed_data_dir: str = Field(default="./data/processed")
    model_dir: str = Field(default="./models")
    vncorenlp_dir: str = Field(default=os.path.expanduser("~/.vncorenlp"))
    stopwords_path: str = Field(default="./data/stopwords/vietnamese-stopwords.txt")
    
    # Models
    sbert_model_name: str = Field(default="bkai-foundation-models/vietnamese-bi-encoder")
    cross_encoder_model_name: str = Field(default="itdainb/PhoRanker")
    
    # Retrieval Hyperparameters
    bm25_k1: float = Field(default=1.2)
    bm25_b: float = Field(default=0.65)
    hybrid_alpha: float = Field(default=0.35)
    top_k_retrieval: int = Field(default=15)
    top_k_classify: int = Field(default=15)
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
