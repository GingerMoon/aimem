import hashlib
import uuid
from datetime import datetime

import pytz

from memory.config import MemoryConfig
from infra.embeddings.factory import EmbedderFactory
from infra.utils.llm import parse_messages
from infra.vector_stores.factory import VectorStoreFactory
from memory.mem_item import MemoryItem
from mem_tools.vector.abstract_out_mem import abstract_out_facts
from mem_tools.vector.update_mem import *
from memory.consts import *

logger = logging.getLogger(__name__)


class MemoryVector:
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.embedding_model = EmbedderFactory.create(self.config.embedder.provider, self.config.embedder.config)
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.collection_name = self.config.vector_store.config.collection_name


    def add(self, messages, metadata, filters):
        parsed_messages = parse_messages(messages)

        new_facts = abstract_out_facts(self.llm, parsed_messages)

        new_facts2existing_memories = []
        new_facts_embeddings = {}
        for index, new_fact in enumerate(new_facts):
            fact_embedding = self.embedding_model.embed(new_fact)
            new_facts_embeddings[new_fact] = fact_embedding
            existing_memories = self.vector_store.search(
                query=fact_embedding,
                limit=5,
                filters=filters,
            )
            simplified_existing_memories = [{"id": mem.id, "text": mem.payload["data"]} for mem in existing_memories]
            logging.info(f"existing memories: {existing_memories}")
            new_facts2existing_memories.insert(index, simplified_existing_memories)

        uuid_mapping = [] # elements' order is the same as new_facts elements' order.
        planned_actions = [] # elements' order is the same as new_facts elements' order.
        for index, new_fact in enumerate(new_facts):
            simplified_existing_memories = new_facts2existing_memories[index]

            # mapping UUIDs with integers for handling UUID hallucinations
            temp_uuid_mapping = {}
            for idx, item in enumerate(simplified_existing_memories):
                temp_uuid_mapping[str(idx)] = item["id"]
                item["id"] = str(idx)
            uuid_mapping.insert(index, temp_uuid_mapping)

            new_fact_planed_actions = llm_plan_update_mem(self.llm, simplified_existing_memories, [new_fact])
            if(len(new_fact_planed_actions) == 0):
                logger.error(f"update_mem_vec returns empty.")
            else:
                logger.info(f"update_mem_vec:\n{new_fact_planed_actions}, ")
                planned_actions.insert(index, new_fact_planed_actions["memory"])

        returned_memories = []
        try:
            for index, actions in enumerate(planned_actions):
                for action in actions:
                    logging.info(action)
                    try:
                        if action[OPERATION_TYPE] == "ADD":
                            memory_id = self._create_memory(
                                data=action["text"], existing_embeddings=new_facts_embeddings, metadata=metadata, filters=filters
                            )
                            returned_memories.append(
                                {
                                    "id": memory_id,
                                    "memory": action["text"],
                                    OPERATION_TYPE: action[OPERATION_TYPE],
                                }
                            )
                        elif action[OPERATION_TYPE] == "UPDATE":
                            self._update_memory(
                                memory_id=uuid_mapping[index][action["id"]],
                                data=action["text"],
                                existing_embeddings=new_facts_embeddings,
                                metadata=metadata,
                            )
                            returned_memories.append(
                                {
                                    "id": uuid_mapping[index][action["id"]],
                                    "memory": action["text"],
                                    OPERATION_TYPE: action[OPERATION_TYPE],
                                    OPERATION_PRE_VALUE: action[OPERATION_PRE_VALUE],
                                }
                            )
                        elif action[OPERATION_TYPE] == "DELETE":
                            self._delete_memory(memory_id=uuid_mapping[index][action["id"]])
                            returned_memories.append(
                                {
                                    "id": uuid_mapping[index][action["id"]],
                                    "memory": action["text"],
                                    OPERATION_TYPE: action[OPERATION_TYPE],
                                }
                            )
                        elif action[OPERATION_TYPE] == "NONE":
                            logging.info("NOOP for Memory.")
                    except Exception as e:
                        logging.error(f"Error in planned actions: {index=} {action=} {e}")
        except Exception as e:
            logging.error(f"Error in new_memories_with_actions: {e}")

        return returned_memories


    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            return None

        filters = {key: memory.payload[key] for key in [f"{NAMESPACE}"] if memory.payload.get(key)}

        # Prepare base memory item
        memory_item = MemoryItem(
            id=memory.id,
            memory=memory.payload["data"],
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump(exclude={"score"})

        # Add metadata if there are additional keys
        excluded_keys = {
            "user_id",
            "hash",
            "data",
            "created_at",
            "updated_at",
        }
        additional_metadata = {k: v for k, v in memory.payload.items() if k not in excluded_keys}
        if additional_metadata:
            memory_item["metadata"] = additional_metadata

        result = {**memory_item, **filters}

        return result


    def get_all(self, filters, limit):
        memories = self.vector_store.list(filters=filters, limit=limit)

        excluded_keys = {
            "user_id",
            "hash",
            "data",
            "created_at",
            "updated_at",
        }
        all_memories = [
            {
                **MemoryItem(
                    id=mem.id,
                    memory=mem.payload["data"],
                    hash=mem.payload.get("hash"),
                    created_at=mem.payload.get("created_at"),
                    updated_at=mem.payload.get("updated_at"),
                ).model_dump(exclude={"score"}),
                **{key: mem.payload[key] for key in [f"{NAMESPACE}"] if key in mem.payload},
                **(
                    {"metadata": {k: v for k, v in mem.payload.items() if k not in excluded_keys}}
                    if any(k for k in mem.payload if k not in excluded_keys)
                    else {}
                ),
            }
            for mem in memories[0]
        ]
        return all_memories


    def search(self, query, filters, limit):
        embeddings = self.embedding_model.embed(query)
        memories = self.vector_store.search(query=embeddings, limit=limit, filters=filters)

        excluded_keys = {
            "user_id",
            "hash",
            "data",
            "created_at",
            "updated_at",
        }

        original_memories = [
            {
                **MemoryItem(
                    id=mem.id,
                    memory=mem.payload["data"],
                    hash=mem.payload.get("hash"),
                    created_at=mem.payload.get("created_at"),
                    updated_at=mem.payload.get("updated_at"),
                    score=mem.score,
                ).model_dump(),
                **{key: mem.payload[key] for key in [f"{NAMESPACE}"] if key in mem.payload},
                **(
                    {"metadata": {k: v for k, v in mem.payload.items() if k not in excluded_keys}}
                    if any(k for k in mem.payload if k not in excluded_keys)
                    else {}
                ),
            }
            for mem in memories
        ]

        return original_memories

    def update(self, memory_id, data):
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (dict): Data to update the memory with.

        Returns:
            dict: Updated memory.
        """
        existing_embeddings = {data: self.embedding_model.embed(data)}

        self._update_memory(memory_id, data, existing_embeddings)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        self._delete_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, filters):

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)

        logger.info(f"Deleted {len(memories)} memories")

        return {"message": "Memories deleted successfully!"}

    def _create_memory(self, data, existing_embeddings, metadata=None, filters=None):
        logging.info(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data)
        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        if filters and NAMESPACE in filters:
            metadata[NAMESPACE] = filters[NAMESPACE]

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )
        return memory_id

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")
        prev_value = existing_memory.payload.get("data")

        new_metadata = metadata or {}
        new_metadata["data"] = data
        new_metadata["hash"] = existing_memory.payload.get("hash")
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        if NAMESPACE in existing_memory.payload:
            new_metadata[f"{NAMESPACE}"] = existing_memory.payload[f"{NAMESPACE}"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data)
        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with {memory_id=} with {data=}. {prev_value=}")
        return memory_id

    def _delete_memory(self, memory_id):
        logging.info(f"Deleting memory with {memory_id=}")
        existing_memory = self.vector_store.get(vector_id=memory_id)
        prev_value = existing_memory.payload["data"]
        self.vector_store.delete(vector_id=memory_id)
        return memory_id

    def reset(self):
        """
        Reset the memory store.
        """
        logger.warning("Resetting all memories")
        self.vector_store.delete_col()
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )


if __name__ == "__main__":
    config_dict = {
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
    config = MemoryConfig(**config_dict)
    mem = MemoryVector(config)

    metadata = {f"{USER_NAME}": "wx"}
    filters = {f"{NAMESPACE}": "n1"}

    mem.reset()

    content = "My only daughter's name is Hancy. Hancy likes playing football. I work for Giant Network Inc.."
    messages = [{"role": "user", "content": content}]
    memories = mem.add(messages, metadata, filters)

    content = "Hancy doesn't likes playing football. Giant Network Inc. is located in SongJiang District Shanghai China."
    messages = [{"role": "user", "content": content}]
    memories = mem.add(messages, metadata, filters)
    logger.info(f"{memories=}")

    for m in memories:
        stored_mem = mem.get(m["id"])
        logger.info(f"{stored_mem=}")

    mem.delete(memories[0]["id"])
    mem.update(memories[1]["id"], memories[0]["memory"] + "--updated.")
