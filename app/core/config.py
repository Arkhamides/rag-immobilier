from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    openai_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "anthropic/claude-sonnet-4.5"

    documents_path: str = "documents"
    data_path: str = "data"

    max_tools_per_plan: int = 5
    ocr_confidence_threshold: float = 0.7


settings = Settings()
