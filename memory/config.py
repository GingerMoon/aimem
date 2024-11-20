from pydantic import BaseModel, Field

from infra.embeddings.configs import EmbedderConfig
from infra.graphs.configs import GraphStoreConfig
from infra.llms.configs import LlmConfig
from infra.vector_stores.configs import VectorStoreConfig


class MemoryConfig(BaseModel):
    vector_store: VectorStoreConfig = Field(
        description="Configuration for the vector store",
        default_factory=VectorStoreConfig,
    )
    llm: LlmConfig = Field(
        description="Configuration for the language model",
        default_factory=LlmConfig,
    )
    embedder: EmbedderConfig = Field(
        description="Configuration for the embedding model",
        default_factory=EmbedderConfig,
    )
    graph_store: GraphStoreConfig = Field(
        description="Configuration for the graph",
        default_factory=GraphStoreConfig,
    )