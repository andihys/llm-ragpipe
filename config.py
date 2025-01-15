from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        case_sensitive=False,
        extra='ignore'
    )

class AzureSettings(BaseConfig):
    azure_deployment: str = Field(..., env='AZURE_DEPLOYMENT')
    azure_api_version: str = Field(..., env='AZURE_API_VERSION')
    azure_deployment_embedding: str = Field(..., env='AZURE_DEPLOYMENT_EMBEDDING')
    temperature: float = Field(0.0, env='TEMPERATURE')
    top_p: float = Field(1.0, env='TOP_P')

azure_settings = AzureSettings()