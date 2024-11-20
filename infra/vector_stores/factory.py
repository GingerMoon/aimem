from infra.llms.base import BaseLlmConfig
from infra.utils.reflection import load_class


class VectorStoreFactory:
    provider_to_class = {
        "qdrant": "infra.vector_stores.qdrant.qdrant.Qdrant",
        "pgvector": "infra.vector_stores.pgvector.pgvector.PGVector",
        "milvus": "infra.vector_stores.milvus.milvus.MilvusDB",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            if not isinstance(config, dict):
                config = config.model_dump()
            vector_store_instance = load_class(class_type)
            return vector_store_instance(**config)
        else:
            raise ValueError(f"Unsupported VectorStore provider: {provider_name}")
