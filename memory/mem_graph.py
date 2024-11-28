import asyncio

from infra.logutil.context import *
from memory.config import MemoryConfig

try:
    from langchain_community.graphs import Neo4jGraph
except ImportError:
    raise ImportError("langchain_community is not installed. Please install it using pip install langchain-community")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is not installed. Please install it using pip install rank-bm25")

from infra.embeddings.factory import EmbedderFactory
from mem_tools.graph.extract_entities_for_add_mem import extract_entities_for_add_mem
from mem_tools.graph.extract_nodes_for_search import extract_nodes_for_search
from mem_tools.graph.update_mem import update_mem
from mem_tools.graph.plan_mem_op import *
from memory.consts import *


class MemoryGraph:
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config
        self.graph = Neo4jGraph(
            self.config.graph_store.config.url,
            self.config.graph_store.config.username,
            self.config.graph_store.config.password,
        )
        self.embedding_model = EmbedderFactory.create(self.config.embedder.provider, self.config.embedder.config)

        self.llm_provider = "openai_structured"
        if self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store.llm:
            self.llm_provider = self.config.graph_store.llm.provider

        self.llm = LlmFactory.create(self.llm_provider, self.config.llm.config)
        self.threshold = 0.7

    async def _graphdb_add(self, namespace, operation):

        source = operation[SOURCE_NODE].lower().replace(" ", "_")
        source_type = operation[SOURCE_TYPE][0].lower().replace(" ", "_")
        relation = operation[RELATION].lower().replace(" ", "_")
        destination = operation[DESTINATION_NODE].lower().replace(" ", "_")
        destination_type = operation[DESTINATION_TYPE][0].lower().replace(" ", "_")
        logging.debug(f"add relationship: {source} -{relation}-> {destination}")

        # Create embeddings
        source_embedding = self.embedding_model.embed(source)
        dest_embedding = self.embedding_model.embed(destination)

        # Updated Cypher query to include node types and embeddings
        cypher = f"""
            MERGE (n:{source_type} {{name: $source_name, {NAMESPACE}: ${NAMESPACE}}})
            ON CREATE SET n.created = timestamp(), n.embedding = $source_embedding
            ON MATCH SET n.embedding = $source_embedding
            MERGE (m:{destination_type} {{name: $dest_name, {NAMESPACE}: ${NAMESPACE}}})
            ON CREATE SET m.created = timestamp(), m.embedding = $dest_embedding
            ON MATCH SET m.embedding = $dest_embedding
            MERGE (n)-[rel:{relation}]->(m)
            ON CREATE SET rel.created = timestamp()
            RETURN n, rel, m
            """

        params = {
            "source_name": source,
            "dest_name": destination,
            "source_embedding": source_embedding,
            "dest_embedding": dest_embedding,
            f"{NAMESPACE}": namespace,
        }

        _ = self.graph.query(cypher, params=params)

    async def _graphdb_del(self, namespace, operation):
        source = operation[SOURCE_NODE].lower().replace(" ", "_")
        relation = operation[RELATION].lower().replace(" ", "_")
        destination = operation[DESTINATION_NODE].lower().replace(" ", "_")
        logging.debug(f"delete relationship: {source} -{relation}-> {destination}")

        # Delete any existing relationship between the nodes
        delete_query = f"""
        MATCH (s {{name: $source, {NAMESPACE}: ${NAMESPACE}}})-[r]->(d {{name: $target, {NAMESPACE}: ${NAMESPACE}}})
        DELETE r
        WITH s,d
        WHERE COUNT{{(s)--()}} = 0
        DETACH DELETE s
        WITH d
        WHERE COUNT{{(d)--()}} = 0
        DETACH DELETE d
        """
        self.graph.query(
            delete_query,
            params={"source": source, "target": destination, f"{NAMESPACE}": namespace},
        )

    async def add(self, messages, metadata, filters):
        """
        Adds data to the graph.

        Args:
            messages (array): The classic openai API chat completion messages.
            metadata (dict): A dictionary containing metadata, such as user_name=xiang.
            filters (dict): A dictionary containing filters such as namespace=n1.
        """
        user_name = metadata[f"{USER_NAME}"]
        namespace = filters[f"{NAMESPACE}"]

        extend_log_ctx_tags({LOG_FLOWNAME: LOG_FLOWNAME_GRAPH_MEM_ADD})

        new_memories = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])

        existing_memories = await self._search(new_memories, metadata, filters)
        logging.info(f"{json.dumps(existing_memories)=}")
        mem_operations = plan_mem_operations(self.llm, user_name, existing_memories, new_memories)
        logging.info(f"{json.dumps(mem_operations)=}")

        add_operations = []
        del_operations = []
        for operation in mem_operations:
            if operation[MEM_OPERATION] == MEM_OP_ADD:
                add_operations.append(operation)
            if operation[MEM_OPERATION] == MEM_OP_DELETE:
                del_operations.append(operation)

        del_tasks = [self._graphdb_del(namespace, del_op) for del_op in del_operations]
        add_tasks = [self._graphdb_add(namespace, add_op) for add_op in add_operations]
        tasks = del_tasks + add_tasks
        await asyncio.gather(*tasks)

        return {"add_operations": add_operations, "del_operations": del_operations}

    async def add_old(self, messages, metadata, filters):
        """
        Adds data to the graph.

        Args:
            messages (array): The classic openai API chat completion messages.
            metadata (dict): A dictionary containing metadata, such as user_name=xiang.
            filters (dict): A dictionary containing filters such as namespace=n1.
        """
        extend_log_ctx_tags({LOG_FLOWNAME: LOG_FLOWNAME_GRAPH_MEM_ADD})

        data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])

        search_task = asyncio.create_task(self._search(data, metadata, filters))
        extract_entities_task = asyncio.create_task(extract_entities_for_add_mem(metadata[USER_NAME], self.llm, data))
        existing_srds, srds_from_input = await asyncio.gather(search_task, extract_entities_task)
        logging.info(f"{json.dumps(existing_srds)=}")
        logging.info(f"{json.dumps(srds_from_input)=}")

        # retrieve the search results
        # search_output = self._search(data, metadata, filters)
        #
        # extracted_entities = extract_entities_for_add_mem(metadata[USER_NAME], self.llm, data)
        # logging.info(f"Extracted entities: {extracted_entities}")

        tasks = [update_mem(self.llm, existing_srds, srd_from_input) for srd_from_input in srds_from_input]
        results = await asyncio.gather(*tasks)

        to_be_added = []
        to_be_updated = []
        for a, u in results:
            to_be_added.extend(a)
            to_be_updated.extend(u)

        logging.info(f"{to_be_added=}")
        logging.info(f"{to_be_updated=}")

        tasks = [self._update_relationship(
            item["source"],
            item["destination"],
            item["relationship"],
            filters,
        ) for item in to_be_updated]
        await asyncio.gather(*tasks)

        async def add_items(item):
            source = item["source"].lower().replace(" ", "_")
            source_type = item["source_type"].lower().replace(" ", "_")
            relation = item["relationship"].lower().replace(" ", "_")
            destination = item["destination"].lower().replace(" ", "_")
            destination_type = item["destination_type"].lower().replace(" ", "_")

            # Create embeddings
            source_embedding = self.embedding_model.embed(source)
            dest_embedding = self.embedding_model.embed(destination)

            # Updated Cypher query to include node types and embeddings
            cypher = f"""
            MERGE (n:{source_type} {{name: $source_name, {NAMESPACE}: ${NAMESPACE}}})
            ON CREATE SET n.created = timestamp(), n.embedding = $source_embedding
            ON MATCH SET n.embedding = $source_embedding
            MERGE (m:{destination_type} {{name: $dest_name, {NAMESPACE}: ${NAMESPACE}}})
            ON CREATE SET m.created = timestamp(), m.embedding = $dest_embedding
            ON MATCH SET m.embedding = $dest_embedding
            MERGE (n)-[rel:{relation}]->(m)
            ON CREATE SET rel.created = timestamp()
            RETURN n, rel, m
            """

            params = {
                "source_name": source,
                "dest_name": destination,
                "source_embedding": source_embedding,
                "dest_embedding": dest_embedding,
                f"{NAMESPACE}": filters[f"{NAMESPACE}"],
            }

            _ = self.graph.query(cypher, params=params)


        tasks = [add_items(item) for item in to_be_added]
        await asyncio.gather(*tasks)

        return {"to_be_added": to_be_added, "to_be_updated": to_be_updated}

    async def _search(self, query, metadata, filters, limit=100):
        nodes_from_query = extract_nodes_for_search(metadata[f"{USER_NAME}"], self.llm, query)
        logging.info(f"{nodes_from_query=}")
        nodes_from_query_set = set(nodes_from_query)

        existing_srds = []
        node_list_hop1 = set()

        cypher_query = f"""
            MATCH (n)
            WHERE n.embedding IS NOT NULL AND n.{NAMESPACE} = ${NAMESPACE}
            WITH n,
                round(reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) | dot + n.embedding[i] * $n_embedding[i]) /
                (sqrt(reduce(l2 = 0.0, i IN range(0, size(n.embedding)-1) | l2 + n.embedding[i] * n.embedding[i])) *
                sqrt(reduce(l2 = 0.0, i IN range(0, size($n_embedding)-1) | l2 + $n_embedding[i] * $n_embedding[i]))), 4) AS similarity
            WHERE similarity >= $threshold
            MATCH (n)-[r]->(m)
            RETURN n.name AS source, labels(n) AS source_type, elementId(n) AS source_id, type(r) AS relation, elementId(r) AS relation_id, m.name AS destination, labels(m) AS destination_type, elementId(m) AS destination_id, similarity
            UNION
            MATCH (n)
            WHERE n.embedding IS NOT NULL AND n.{NAMESPACE} = ${NAMESPACE}
            WITH n,
                round(reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) | dot + n.embedding[i] * $n_embedding[i]) /
                (sqrt(reduce(l2 = 0.0, i IN range(0, size(n.embedding)-1) | l2 + n.embedding[i] * n.embedding[i])) *
                sqrt(reduce(l2 = 0.0, i IN range(0, size($n_embedding)-1) | l2 + $n_embedding[i] * $n_embedding[i]))), 4) AS similarity
            WHERE similarity >= $threshold
            MATCH (m)-[r]->(n)
            RETURN m.name AS source, labels(m) AS source_type, elementId(m) AS source_id, type(r) AS relation, elementId(r) AS relation_id, n.name AS destination, labels(n) AS destination_type, elementId(n) AS destination_id, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """

        for node in nodes_from_query:
            n_embedding = self.embedding_model.embed(node)
            params = {
                "n_embedding": n_embedding,
                "threshold": self.threshold,
                f"{NAMESPACE}": filters[f"{NAMESPACE}"],
                "limit": limit,
            }
            # srd : source_relation_destination
            srds = self.graph.query(cypher_query, params=params)
            existing_srds.extend(srds)
            for srd in srds:
                s = srd["source"]
                d = srd["destination"]
                if s not in nodes_from_query_set:
                    node_list_hop1.add(s)
                if d not in nodes_from_query_set:
                    node_list_hop1.add(d)

        logging.info(f"{json.dumps(existing_srds)=}")

        logging.info(f"{node_list_hop1=}")
        for node in node_list_hop1:
            n_embedding = self.embedding_model.embed(node)
            params = {
                "n_embedding": n_embedding,
                "threshold": self.threshold,
                f"{NAMESPACE}": filters[f"{NAMESPACE}"],
                "limit": limit,
            }
            srds = self.graph.query(cypher_query, params=params)
            existing_srds.extend(srds)

        logging.info(f"{json.dumps(existing_srds)=}")
        return existing_srds

    async def search(self, query, metadata, filters, limit=100):
        """
        Search for memories and related graph data.

        Args:
            query (str): Query to search for.
            metadata (dict): A dictionary containing metadata, such as user_name=xiang.
            filters (dict): A dictionary containing filters to be applied during the search.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.

        Returns:
            dict: A dictionary containing:
                - "contexts": List of search results from the base data store.
                - "entities": List of related graph data based on the query.
        """

        extend_log_ctx_tags({LOG_FLOWNAME: LOG_FLOWNAME_GRAPH_MEM_SEARCH})

        search_output = await self._search(query, metadata, filters, limit)

        if not search_output:
            return []

        search_outputs_sequence = [[item["source"], item["relation"], item["destination"]] for item in search_output]
        # search_outputs_sequence deduplication
        srd_json_str_set = set()
        srds = []
        for srd in search_outputs_sequence:
            json_string = json.dumps(srd)
            if json_string not in srd_json_str_set:
                srd_json_str_set.add(json_string)
                srds.append(srd)
        search_outputs_sequence = srds

        bm25 = BM25Okapi(search_outputs_sequence)

        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=5)

        search_results = []
        for item in reranked_results:
            search_results.append({"source": item[0], "relationship": item[1], "target": item[2]})

        logging.info(f"{search_results=}")

        return search_results

    def delete_all(self, filters):
        cypher = f"""
        MATCH (n {{{NAMESPACE}: ${NAMESPACE}}})
        DETACH DELETE n
        """
        params = {f"{NAMESPACE}": filters[f"{NAMESPACE}"]}
        self.graph.query(cypher, params=params)

    async def get_all(self, filters, limit=100):
        """
        Retrieves all nodes and relationships from the graph database based on optional filtering criteria.

        Args:
            filters (dict): A dictionary containing filters to be applied during the retrieval.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.
        Returns:
            list: A list of dictionaries, each containing:
                - 'contexts': The base data store response for each memory.
                - 'entities': A list of strings representing the nodes and relationships
        """

        # return all nodes and relationships
        query = f"""
        MATCH (n {{{NAMESPACE}: ${NAMESPACE}}})-[r]->(m {{{NAMESPACE}: ${NAMESPACE}}})
        RETURN n.name AS source, type(r) AS relationship, m.name AS target
        LIMIT $limit
        """
        results = self.graph.query(query, params={f"{NAMESPACE}": filters[f"{NAMESPACE}"], "limit": limit})

        final_results = []
        for result in results:
            final_results.append(
                {
                    "source": result["source"],
                    "relationship": result["relationship"],
                    "target": result["target"],
                }
            )

        logging.info(f"Retrieved {len(final_results)} relationships")

        return final_results

    async def _update_relationship(self, source, target, relationship, filters):
        """
        Update or create a relationship between two nodes in the graph.

        Args:
            source (str): The name of the source node.
            target (str): The name of the target node.
            relationship (str): The type of the relationship.
            filters (dict): A dictionary containing filter such as namespace=n1

        Raises:
            Exception: If the operation fails.
        """
        logging.info(f"Updating relationship: {source} -{relationship}-> {target}")

        relationship = relationship.lower().replace(" ", "_")

        # Check if nodes exist and create them if they don't
        check_and_create_query = f"""
        MERGE (n1 {{name: $source, {NAMESPACE}: ${NAMESPACE}}})
        MERGE (n2 {{name: $target, {NAMESPACE}: ${NAMESPACE}}})
        """
        self.graph.query(
            check_and_create_query,
            params={"source": source, "target": target, f"{NAMESPACE}": filters[f"{NAMESPACE}"]},
        )

        # Delete any existing relationship between the nodes
        delete_query = f"""
        MATCH (n1 {{name: $source, {NAMESPACE}: ${NAMESPACE}}})-[r]->(n2 {{name: $target, {NAMESPACE}: ${NAMESPACE}}})
        DELETE r
        """
        self.graph.query(
            delete_query,
            params={"source": source, "target": target, f"{NAMESPACE}": filters[f"{NAMESPACE}"]},
        )

        # Create the new relationship
        create_query = f"""
        MATCH (n1 {{name: $source, {NAMESPACE}: ${NAMESPACE}}}), (n2 {{name: $target, {NAMESPACE}: ${NAMESPACE}}})
        CREATE (n1)-[r:{relationship}]->(n2)
        RETURN n1, r, n2
        """
        result = self.graph.query(
            create_query,
            params={"source": source, "target": target, f"{NAMESPACE}": filters[f"{NAMESPACE}"]},
        )

        if not result:
            raise Exception(f"Failed to update or create relationship between {source} and {target}")

async def __test__():
    config_dict = {
        "version": "v1.1",
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
    config = MemoryConfig(**config_dict)

    metadata = {f"{USER_NAME}": "wx"}
    filters = {f"{NAMESPACE}": "n1"}

    mem = MemoryGraph(config)

    mem.delete_all(filters)

    # content = "My only pet's name is Meow. Hancy likes playing football. Giant Network Inc. is located in SongJiang District Shanghai China. I like playing football."
    # messages = [{"role": "user", "content": content}]
    # memories = await mem.add(messages, metadata, filters)
    # logging.info(f"{memories=}")
    #
    # content = "My pet's name is Kitty. Hancy doesn't likes playing football. Giant Network Inc. is located in BaoShan District."
    # messages = [{"role": "user", "content": content}]
    # memories = await mem.add(messages, metadata, filters)
    # logging.info(f"{memories=}")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hi, can you tell me about the weather in New York today?"
        },
        {
            "role": "assistant",
            "content": "Hello! The weather in New York today is partly cloudy with a high of 72°F and a low of 60°F. There's a 10% chance of rain, so you might want to carry an umbrella just in case."
        },
        {
            "role": "user",
            "content": "Thanks! What’s the best way to get from Manhattan to Brooklyn?"
        },
        {
            "role": "assistant",
            "content": "The best way to get from Manhattan to Brooklyn depends on your starting point and destination. Generally, you can use the subway (e.g., A, C, F, or R lines), take a taxi, or even bike across the Brooklyn Bridge if you're up for a scenic ride. Which method would you prefer?"
        },
        {
            "role": "user",
            "content": "I think I’ll take the subway. Oh, and do you know any good pizza places in Brooklyn?"
        },
        {
            "role": "assistant",
            "content": "Great choice! The subway is very convenient. For pizza in Brooklyn, you should definitely check out Roberta's in Bushwick, Di Fara Pizza in Midwood, and Grimaldi's under the Brooklyn Bridge. Each has its own unique style and is highly recommended!"
        },
        {
            "role": "user",
            "content": "Awesome, thanks! By the way, do you like cats or dogs more?"
        },
        {
            "role": "assistant",
            "content": "As an AI, I don't have personal preferences, but I can appreciate both cats and dogs for their unique qualities. Cats are often independent and can be quite charming, while dogs are known for their loyalty and energy. Do you have a favorite?"
        },
        {
            "role": "user",
            "content": "I’m more of a dog person. They’re just so lovable and fun to play with. But my roommate loves cats. We always argue about it."
        },
        {
            "role": "assistant",
            "content": "It sounds like a classic debate! Both pets have their pros and cons, and it's great that you and your roommate can have some fun with it. Maybe you could compromise and adopt a cat and a dog together? Just a thought! 😊"
        },
        {
            "role": "user",
            "content": "Haha, maybe one day! Speaking of pets, what’s the capital of France?"
        },
        {
            "role": "assistant",
            "content": "The capital of France is Paris. It's a beautiful city known for its art, culture, and iconic landmarks like the Eiffel Tower and Notre-Dame Cathedral. Is there anything else you'd like to know about Paris or France?"
        },
        {
            "role": "user",
            "content": "No, that’s all. Thanks for the help! Have a nice day!"
        },
        {
            "role": "assistant",
            "content": "You're welcome! If you have any more questions, feel free to ask. Have a great day too!"
        }
    ]
    memories = await mem.add(messages, metadata, filters)
    logging.info(f"{memories=}")


if __name__ == "__main__":
    asyncio.run(__test__())


