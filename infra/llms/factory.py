from infra.llms.base import BaseLlmConfig

from infra.utils.reflection import load_class


class LlmFactory:
    provider_to_class = {
        "aliyun": "infra.llms.aliyun.DashscopeLlm",
        "ollama": "infra.llms.ollama.OllamaLLM",
        "openai": "infra.llms.openai.OpenAILLM",
        "litellm": "infra.llms.litellm.LiteLLM",
        "openai_structured": "infra.llms.openai_structured.OpenAIStructuredLLM",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            llm_instance = load_class(class_type)
            base_config = BaseLlmConfig(**config)
            return llm_instance(base_config)
        else:
            raise ValueError(f"Unsupported Llm provider: {provider_name}")
