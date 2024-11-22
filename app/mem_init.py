import os

from memory.mem import Memory

config_dict = {
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "neo4j://localhost:7687",
            "username": "neo4j",
            "password": "xxx"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "test",
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 1024,  # Change this according to your local model's dimensions
        },
    },
    "llm": {
        "provider": "aliyun",
        "config": {
            "api_key": os.environ.get("DASHSCOPE_API_KEY"),
            "model": "qwen-max",
            "temperature": 0.00001,
            "top_p": 0.00001,
            "max_tokens": 1500,
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "text-embedding-v3",
        },
    },
}
mem = Memory.from_config(config_dict)