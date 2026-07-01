from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str
    llm_model: str = "claude-sonnet-4-6"

    documents_path: str = "documents"
    data_path: str = "data"

    max_tools_per_plan: int = 5
    ocr_confidence_threshold: float = 0.7

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"


settings = Settings()
