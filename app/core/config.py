from functools import lru_cache
from urllib.parse import quote

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Dodo Agent Python"
    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8888
    serve_frontend: bool = True
    cors_origins: str = "*"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.deepseek.com"
    chat_model: str = "deepseek-v4-flash"
    chat_temperature: float = 0.5
    model_timeout_seconds: float = 120
    agent_recursion_limit: int = 24
    tavily_api_key: str = ""
    tavily_mcp_url: str = "https://mcp.tavily.com/mcp/"
    tavily_request_timeout_seconds: float = 300
    mysql_url: str = (
        "jdbc:mysql://127.0.0.1:3307/dodo?useUnicode=true&characterEncoding=utf8"
        "&useSSL=false&allowPublicKeyRetrieval=true&serverTimezone=GMT%2B8"
    )
    mysql_username: str = "root"
    mysql_password: str = ""
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_database: int = 0
    redis_password: str = ""
    embedding_api_key: str = ""
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/"
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = 1024
    minio_endpoint: str = "http://127.0.0.1:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket_name: str = "rag-test2"
    pgvector_url: str = ""
    pgvector_host: str = "127.0.0.1"
    pgvector_port: int = 5432
    pgvector_database: str = "vector_store"
    pgvector_user: str = "postgres"
    pgvector_password: str = ""
    vision_api_key: str = ""
    vision_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/"
    vision_model: str = "qwen3-vl-plus"
    max_upload_bytes: int = 50 * 1024 * 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def resolved_pgvector_url(self) -> str:
        """优先返回兼容 URL；否则用 Java 同名的 PGVector 分项配置构造异步连接串。"""
        if self.pgvector_url.strip():
            return self.pgvector_url.strip()
        user = quote(self.pgvector_user, safe="")
        password = quote(self.pgvector_password, safe="")
        credentials = user if not password else f"{user}:{password}"
        return (
            f"postgresql+psycopg://{credentials}@{self.pgvector_host}:"
            f"{self.pgvector_port}/{self.pgvector_database}"
        )

    def parsed_cors_origins(self) -> list[str]:
        origins = [origin.strip() for origin in self.cors_origins.split(",")]
        return [origin for origin in origins if origin] or ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
