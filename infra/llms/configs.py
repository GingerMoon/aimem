from typing import Optional

from pydantic import BaseModel, Field, field_validator
from infra.llms.factory import provider_to_class

class LlmConfig(BaseModel):
    provider: str = Field(description="Provider of the LLM (e.g., 'ollama', 'openai')", default="openai")
    config: Optional[dict] = Field(description="Configuration for the specific LLM", default={})

    @field_validator("config")
    def validate_config(cls, v, values):
        provider = values.data.get("provider")
        if provider in provider_to_class.keys():
            return v
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
