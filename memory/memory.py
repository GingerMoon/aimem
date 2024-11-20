import concurrent
import hashlib
import json
import logging
import os
import uuid
import warnings
from datetime import datetime
from typing import Any, Dict

import pytz
from pydantic import ValidationError

from config import MemoryConfig
from base import MemoryBase
from infra.llms.factory import LlmFactory
from infra.embeddings.factory import EmbedderFactory
from infra.vector_stores.factory import VectorStoreFactory
from infra.utils.llm import parse_messages
from mem_tools.abstract_out_mem import abstract_out_mem
from mem_tools.vector.update_mem import update_mem as update_mem_vec
from memory_item import MemoryItem

logger = logging.getLogger(__name__)


class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.embedding_model = EmbedderFactory.create(self.config.embedder.provider, self.config.embedder.config)
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.collection_name = self.config.vector_store.config.collection_name

        self.enable_graph = False

        if self.config.graph_store.config:
            from graph_memory import MemoryGraph

            self.graph = MemoryGraph(self.config)
            self.enable_graph = True

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    def add(
        self,
        messages,
        user_id=None,
        metadata=None,
        filters=None,
    ):
        """
        Create a new memory.

        Args:
            messages (str or List[Dict[str, str]]): Messages to store in the memory.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.

        Returns:
            dict: A dictionary containing the result of the memory addition operation.
            result: dict of affected events with each dict has the following key:
              'memories': affected memories
              'graph': affected graph memories

              'memories' and 'graph' is a dict, each with following subkeys:
                'add': added memory
                'update': updated memory
                'delete': deleted memory


        """

        if metadata is None:
            metadata = {}
        if filters is None:
            filters = {}

        metadata["user_id"] = user_id
        filters["user_id"] = user_id

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(self._add_to_vector_store, messages, filters)
            future2 = executor.submit(self._add_to_graph, messages, filters)

            concurrent.futures.wait([future1, future2])

            vector_store_result = future1.result()
            graph_result = future2.result()

        return {
            "vector_memory": vector_store_result,
            "graph_memory": graph_result,
        }

    def _add_to_vector_store(self, messages, metadata, filters):
        parsed_messages = parse_messages(messages)

        new_retrieved_facts = abstract_out_mem(self.llm, parsed_messages)

        dict_new_facts_to_existing_memories = []
        new_message_embeddings = {}
        for index, new_mem in enumerate(new_retrieved_facts):
            messages_embeddings = self.embedding_model.embed(new_mem)
            new_message_embeddings[new_mem] = messages_embeddings
            existing_memories = self.vector_store.search(
                query=messages_embeddings,
                limit=5,
                filters=filters,
            )
            simplified_existing_memories = [{"id": mem.id, "text": mem.payload["data"]} for mem in existing_memories]
            logging.info(f"existing memories: {existing_memories}")
            dict_new_facts_to_existing_memories.insert(index, simplified_existing_memories)


        new_memories_with_actions = []
        for index, new_retrieved_fact in enumerate(new_retrieved_facts):
            simplified_existing_memories = dict_new_facts_to_existing_memories[index]

            # mapping UUIDs with integers for handling UUID hallucinations
            temp_uuid_mapping = {}
            for idx, item in enumerate(simplified_existing_memories):
                temp_uuid_mapping[str(idx)] = item["id"]
                item["id"] = str(idx)

            counter = 0
            mem_with_actions = update_mem_vec(self.llm, simplified_existing_memories, [new_retrieved_fact])
            while(len(mem_with_actions) == 0 and counter < 5):
                counter += 1
                mem_with_actions = update_mem_vec(self.llm, simplified_existing_memories, [new_retrieved_fact])
            if(len(mem_with_actions) == 0):
                logger.error(f"update_mem_vec returns empty after {counter} retries")
            else:
                logger.info(f"update_mem_vec:\n{mem_with_actions}, ")
                new_memories_with_actions.extend(mem_with_actions["memory"])


        # # mapping UUIDs with integers for handling UUID hallucinations
        # temp_uuid_mapping = {}
        # for idx, item in enumerate(retrieved_old_memory):
        #     temp_uuid_mapping[str(idx)] = item["id"]
        #     retrieved_old_memory[idx]["id"] = str(idx)
        #
        # counter = 0
        # new_memories_with_actions = update_mem_vec(self.llm, retrieved_old_memory, new_retrieved_facts)
        # while(len(new_memories_with_actions) == 0):
        #     counter += 1
        #     new_memories_with_actions = update_mem_vec(self.llm, existing_memories, new_retrieved_facts)

        returned_memories = []
        try:
            for resp in new_memories_with_actions:
                logging.info(resp)
                try:
                    if resp["event"] == "ADD":
                        memory_id = self._create_memory(
                            data=resp["text"], existing_embeddings=new_message_embeddings, metadata=metadata
                        )
                        returned_memories.append(
                            {
                                "id": memory_id,
                                "memory": resp["text"],
                                "event": resp["event"],
                            }
                        )
                    elif resp["event"] == "UPDATE":
                        self._update_memory(
                            memory_id=temp_uuid_mapping[resp["id"]],
                            data=resp["text"],
                            existing_embeddings=new_message_embeddings,
                            metadata=metadata,
                        )
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp["id"]],
                                "memory": resp["text"],
                                "event": resp["event"],
                                "previous_memory": resp["old_memory"],
                            }
                        )
                    elif resp["event"] == "DELETE":
                        self._delete_memory(memory_id=temp_uuid_mapping[resp["id"]])
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp["id"]],
                                "memory": resp["text"],
                                "event": resp["event"],
                            }
                        )
                    elif resp["event"] == "NONE":
                        logging.info("NOOP for Memory.")
                except Exception as e:
                    logging.error(f"Error in new_memories_with_actions: {e}")
        except Exception as e:
            logging.error(f"Error in new_memories_with_actions: {e}")

        return returned_memories

    def _add_to_graph(self, messages, filters):
        added_entities = []
        if self.api_version == "v1.1" and self.enable_graph:
            if filters["user_id"]:
                self.graph.user_id = filters["user_id"]
            elif filters["agent_id"]:
                self.graph.agent_id = filters["agent_id"]
            elif filters["run_id"]:
                self.graph.run_id = filters["run_id"]
            else:
                self.graph.user_id = "USER"
            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            added_entities = self.graph.add(data, filters)

        return added_entities

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

        filters = {key: memory.payload[key] for key in ["user_id", "agent_id", "run_id"] if memory.payload.get(key)}

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
            "agent_id",
            "run_id",
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

    def get_all(self, user_id=None, agent_id=None, run_id=None, limit=100):
        """
        List all memories.

        Returns:
            list: List of all memories.
        """
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._get_all_from_vector_store, filters, limit)
            future_graph_entities = (
                executor.submit(self.graph.get_all, filters, limit)
                if self.api_version == "v1.1" and self.enable_graph
                else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            all_memories = future_memories.result()
            graph_entities = future_graph_entities.result() if future_graph_entities else None

        if self.api_version == "v1.1":
            if self.enable_graph:
                return {"results": all_memories, "relations": graph_entities}
            else:
                return {"results": all_memories}
        else:
            warnings.warn(
                "The current get_all API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return all_memories

    def _get_all_from_vector_store(self, filters, limit):
        memories = self.vector_store.list(filters=filters, limit=limit)

        excluded_keys = {
            "user_id",
            "agent_id",
            "run_id",
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
                **{key: mem.payload[key] for key in ["user_id", "agent_id", "run_id"] if key in mem.payload},
                **(
                    {"metadata": {k: v for k, v in mem.payload.items() if k not in excluded_keys}}
                    if any(k for k in mem.payload if k not in excluded_keys)
                    else {}
                ),
            }
            for mem in memories[0]
        ]
        return all_memories

    def search(self, query, user_id=None, agent_id=None, run_id=None, limit=100, filters=None):
        """
        Search for memories.

        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            agent_id (str, optional): ID of the agent to search for. Defaults to None.
            run_id (str, optional): ID of the run to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Filters to apply to the search. Defaults to None.

        Returns:
            list: List of search results.
        """
        filters = filters or {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not any(key in filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("One of the filters: user_id, agent_id or run_id is required!")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._search_vector_store, query, filters, limit)
            future_graph_entities = (
                executor.submit(self.graph.search, query, filters, limit)
                if self.api_version == "v1.1" and self.enable_graph
                else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            original_memories = future_memories.result()
            graph_entities = future_graph_entities.result() if future_graph_entities else None

        if self.api_version == "v1.1":
            if self.enable_graph:
                return {"results": original_memories, "relations": graph_entities}
            else:
                return {"results": original_memories}
        else:
            warnings.warn(
                "The current get_all API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return original_memories

    def _search_vector_store(self, query, filters, limit):
        embeddings = self.embedding_model.embed(query)
        memories = self.vector_store.search(query=embeddings, limit=limit, filters=filters)

        excluded_keys = {
            "user_id",
            "agent_id",
            "run_id",
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
                **{key: mem.payload[key] for key in ["user_id", "agent_id", "run_id"] if key in mem.payload},
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

    def delete_all(self, user_id=None, agent_id=None, run_id=None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)

        logger.info(f"Deleted {len(memories)} memories")

        if self.api_version == "v1.1" and self.enable_graph:
            self.graph.delete_all(filters)

        return {"message": "Memories deleted successfully!"}

    def _create_memory(self, data, existing_embeddings, metadata=None):
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

        if "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data)
        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")
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

    def chat(self, system_prompt, query, user_id, run_id):
        search_results = self.search(query, user_id, run_id=run_id, limit=1)
        logging.error(f"search_results: {search_results}")

        facts = []
        for r in search_results["results"]:
            facts.append(r["memory"])
        if len(facts) > 0:
            system_prompt = system_prompt + "\n facts: \n" + json.dumps(facts, ensure_ascii=False)

        if 'relations' in search_results:
            system_prompt = system_prompt + "\n graph entities relations: \n" + json.dumps(search_results["relations"], ensure_ascii=False)

        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            # response_format={"type": "json_object"},
        )
        # data = json.loads(response)
        # if 'answer' in data:
        #     return data['answer']
        # if '回答' in data:
        #     return data['回答']
        return response

if __name__ == "__main__":
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
            "provider": "openai",
            "config": {
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen-max",
                "temperature": 0.02,
                "top_p": 0.02,
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

    content = "My only daughter's name is Hancy. Hancy likes playing football. I work for Giant Network Inc.."
    messages = [{"role": "user", "content": content}]
    metadata = filters = {"user_id": "wx"}
    memories = mem._add_to_vector_store(messages, metadata, filters)

    content = "Hancy doesn't likes playing football. Giant Network Inc. is located in SongJiang District Shanghai China."
    messages = [{"role": "user", "content": content}]
    metadata = filters = {"user_id": "wx"}
    memories = mem._add_to_vector_store(messages, metadata, filters)

    logger.info(memories)