from infra.embeddings.base import BaseEmbedderConfig
from infra.utils.reflection import load_class


class EmbedderFactory:
    provider_to_class = {
        "openai": "infra.embeddings.openai.OpenAIEmbedding",
        "ollama": "infra.embeddings.ollama.OllamaEmbedding",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            embedder_instance = load_class(class_type)
            base_config = BaseEmbedderConfig(**config)
            return embedder_instance(base_config)
        else:
            raise ValueError(f"Unsupported Embedder provider: {provider_name}")