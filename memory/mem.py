import concurrent
from typing import Any, Dict

from pydantic import ValidationError

from base import MemoryBase
from config import MemoryConfig
from mem_tools.vector.update_mem import *
from memory.mem_vec import MemoryVector
from consts import *

logger = logging.getLogger(__name__)


class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)

        self.vec_mem = MemoryVector(self.config)

        self.enable_graph = False

        if self.config.graph_store.config:
            from mem_graph import MemoryGraph

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
        user_name,
        namespace,
        metadata=None,
        filters=None,
    ):
        """
        Create a new memory.

        Args:
            messages (str or List[Dict[str, str]]): Messages to store in the memory.
            metadata (dict): A dictionary containing metadata, such as user_name=xiang.
            filters (dict): A dictionary containing filters such as namespace=n1.

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

        metadata[USER_NAME] = user_name
        filters[NAMESPACE] = namespace

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(self.vec_mem.add, messages, metadata, filters)
            future2 = executor.submit(self.graph.add, messages, metadata, filters)

            concurrent.futures.wait([future1, future2])

            vector_store_result = future1.result()
            graph_result = future2.result()

        return {
            "vector_memory": vector_store_result,
            "graph_memory": graph_result,
        }


    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        return self.vec_mem.get(memory_id)

    def get_all(self, namespace=None, limit=100):
        """
        List all memories.

        Returns:
            list: List of all memories.
        """
        filters = {}
        if namespace:
            filters[NAMESPACE] = namespace

        with concurrent.futures.ThreadPoolExecutor() as executor:
            vec_memories_future = executor.submit(self._get_all_from_vector_store, filters, limit)
            graph_memories_future = (
                executor.submit(self.graph.get_all, filters, limit)
                if self.enable_graph
                else None
            )

            concurrent.futures.wait(
                [vec_memories_future, graph_memories_future] if graph_memories_future else [vec_memories_future]
            )

            all_memories = vec_memories_future.result()
            graph_entities = graph_memories_future.result() if graph_memories_future else None

        if self.enable_graph:
            return {"results": all_memories, "relations": graph_entities}
        else:
            return {"results": all_memories}


    def search(self, query, user_name, namespace, limit=100, filters=None):
        """
        Search for memories.

        Args:
            query (str): Query to search for.
            namespace (str, optional): ID of the namespace to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Filters to apply to the search. Defaults to None.

        Returns:
            list: List of search results.
        """
        filters = filters or {}
        if namespace:
            filters[NAMESPACE] = namespace
        else:
            raise ValueError(f"{NAMESPACE} is required!")

        # graph search will use llm planning tool to extract entities from query, which needs user name.
        metadata = {USER_NAME: user_name}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            vec_memories_future = executor.submit(self.vec_mem.search, query, filters, limit)
            graph_memories_future = (
                executor.submit(self.graph.search, query, metadata, filters, limit)
                if self.enable_graph
                else None
            )

            concurrent.futures.wait(
                [vec_memories_future, graph_memories_future] if graph_memories_future else [vec_memories_future]
            )

            original_memories = vec_memories_future.result()
            graph_entities = graph_memories_future.result() if graph_memories_future else None

        if self.enable_graph:
            return {"results": original_memories, "relations": graph_entities}
        else:
            return {"results": original_memories}


    def update(self, memory_id, data):
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (dict): Data to update the memory with.

        Returns:
            dict: Updated memory.
        """
        self.vec_mem.update(memory_id, data)

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        self.vec_mem.delete(memory_id)

    def delete_all(self, namespace=None):
        """
        Delete all memories.

        Args:
            namespace (str, optional): ID of the user to delete memories for. Defaults to None.
        """
        filters = {}
        if namespace:
            filters[NAMESPACE] = namespace

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        self.vec_mem.delete_all(filters)

        if self.enable_graph:
            self.graph.delete_all(filters)

        return {"message": "Memories deleted successfully!"}


    def reset(self):
        """
        Reset the memory store.
        """
        self.vec_mem.reset()

    def chat(self, system_prompt, query, user_name, namespace):
        search_results = self.search(query, user_name, namespace, limit=1)
        logging.info(f"search_results: {search_results}")

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

    user_name = "wx"
    namespace = "n1"

    mem.delete_all(namespace)

    content = "I have only one daughter whose name is Hancy. Shaque Network Inc. is in SongJiang District Shanghai China. Hancy likes playing football. I work for Shaque Network Inc."
    messages = [{"role": "user", "content": content}]
    memories = mem.add(messages, user_name, namespace)

    content = "Hancy doesn't like football. Shaque Network Inc. is in BaoShan District Shanghai China"
    messages = [{"role": "user", "content": content}]
    memories = mem.add(messages, user_name, namespace)

    logger.info(memories)

    query = "Do you know where am I working?"
    # search_result = mem.search(query, user_name, namespace, 1)
    # logger.info(f"{search_result=}")

    system_prompt = "你是wx的数字人分身.你拥有理解graph entities relations的能力.请基于你所知道的facts和graph entities relations来回答问题.在回答问题时，尽量使用自然、流畅的语言，并且在回答中加入个人的观点和情感，使回答更加生动、有趣。同时，尝试在回答中穿插一些幽默或轻松的元素，让交流更加轻松愉快"

    llm_ans = mem.chat(system_prompt, query, "wx", namespace)
    logger.info(f"{llm_ans}")