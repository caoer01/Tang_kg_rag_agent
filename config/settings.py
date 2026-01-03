from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """系统配置"""

    # LLM配置
    LLM_API_BASE: str = "http://localhost:8000/v1"  # vllm服务地址
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-8B-Instruct"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2000

    # Milvus配置
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION_TEXT: str = "diabetes_docs"
    MILVUS_COLLECTION_IMAGE: str = "diabetes_images"
    VECTOR_DIM: int = 768  # sentence-transformers维度

    # Neo4j配置
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # Embedding配置
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_BATCH_SIZE: int = 32

    # 文档处理配置
    PDF_DATA_PATH: str = "/data/pdf"
    NEO4J_DATA_PATH: str = "/data/neo4j"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # 检索配置
    TOP_K_SIMPLE: int = 5
    TOP_K_COMPLEX: int = 10
    RERANK_TOP_K: int = 3
    BM25_WEIGHT: float = 0.3
    VECTOR_WEIGHT: float = 0.7

    # 质量评估阈值
    QUALITY_THRESHOLD: float = 0.7
    MAX_RETRY: int = 3

    # 复杂问题判定阈值
    COMPLEXITY_THRESHOLD: float = 0.6

    # 记忆配置
    CHECKPOINT_DIR: str = "./checkpoints"
    USER_STORE_DIR: str = "./user_store"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# 创建必要的目录
os.makedirs(settings.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(settings.USER_STORE_DIR, exist_ok=True)