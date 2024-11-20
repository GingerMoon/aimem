import logging
import os

from config import MemoryConfig

try:
    from langchain_community.graphs import Neo4jGraph
except ImportError:
    raise ImportError("langchain_community is not installed. Please install it using pip install langchain-community")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is not installed. Please install it using pip install rank-bm25")

from infra.llms.factory import LlmFactory
from infra.embeddings.factory import EmbedderFactory
from mem_tools.graph.extract_entities_for_add_mem import extract_entities_for_add_mem
from mem_tools.graph.extract_entities_for_search import extract_entities_for_search
from mem_tools.graph.update_mem import update_mem

logger = logging.getLogger(__name__)


class MemoryGraph:
    def __init__(self, config):
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

    def add(self, data, filters):
        """
        Adds data to the graph.

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
        """

        # retrieve the search results
        search_output = []
        search_output = self._search(data, filters)

        extracted_entities = extract_entities_for_add_mem(filters["user_id"], self.llm, data)
        logger.debug(f"Extracted entities: {extracted_entities}")

        to_be_added = []
        to_be_updated = []

        for e in extracted_entities:
            a, u = update_mem(self.llm, search_output, e)
            to_be_added.extend(a)
            to_be_updated.extend(u)

        for item in to_be_updated:
            self._update_relationship(
                item["source"],
                item["destination"],
                item["relationship"],
                filters,
            )

        returned_entities = []
        for item in to_be_added:
            source = item["source"].lower().replace(" ", "_")
            source_type = item["source_type"].lower().replace(" ", "_")
            relation = item["relationship"].lower().replace(" ", "_")
            destination = item["destination"].lower().replace(" ", "_")
            destination_type = item["destination_type"].lower().replace(" ", "_")

            returned_entities.append({"source": source, "relationship": relation, "target": destination})

            # Create embeddings
            source_embedding = self.embedding_model.embed(source)
            dest_embedding = self.embedding_model.embed(destination)

            # Updated Cypher query to include node types and embeddings
            cypher = f"""
            MERGE (n:{source_type} {{name: $source_name, user_id: $user_id}})
            ON CREATE SET n.created = timestamp(), n.embedding = $source_embedding
            ON MATCH SET n.embedding = $source_embedding
            MERGE (m:{destination_type} {{name: $dest_name, user_id: $user_id}})
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
                "user_id": filters["user_id"],
            }

            _ = self.graph.query(cypher, params=params)

        logger.info(f"Added {len(to_be_added)} new memories to the graph")

        return returned_entities

    def _search(self, query, filters, limit=100):
        node_list = extract_entities_for_search(filters["user_id"], self.llm, query)
        logger.debug(f"Node list for search query : {node_list}")

        result_relations = []

        for node in node_list:
            n_embedding = self.embedding_model.embed(node)

            cypher_query = """
            MATCH (n)
            WHERE n.embedding IS NOT NULL AND n.user_id = $user_id
            WITH n,
                round(reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) | dot + n.embedding[i] * $n_embedding[i]) /
                (sqrt(reduce(l2 = 0.0, i IN range(0, size(n.embedding)-1) | l2 + n.embedding[i] * n.embedding[i])) *
                sqrt(reduce(l2 = 0.0, i IN range(0, size($n_embedding)-1) | l2 + $n_embedding[i] * $n_embedding[i]))), 4) AS similarity
            WHERE similarity >= $threshold
            MATCH (n)-[r]->(m)
            RETURN n.name AS source, elementId(n) AS source_id, type(r) AS relation, elementId(r) AS relation_id, m.name AS destination, elementId(m) AS destination_id, similarity
            UNION
            MATCH (n)
            WHERE n.embedding IS NOT NULL AND n.user_id = $user_id
            WITH n,
                round(reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) | dot + n.embedding[i] * $n_embedding[i]) /
                (sqrt(reduce(l2 = 0.0, i IN range(0, size(n.embedding)-1) | l2 + n.embedding[i] * n.embedding[i])) *
                sqrt(reduce(l2 = 0.0, i IN range(0, size($n_embedding)-1) | l2 + $n_embedding[i] * $n_embedding[i]))), 4) AS similarity
            WHERE similarity >= $threshold
            MATCH (m)-[r]->(n)
            RETURN m.name AS source, elementId(m) AS source_id, type(r) AS relation, elementId(r) AS relation_id, n.name AS destination, elementId(n) AS destination_id, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """
            params = {
                "n_embedding": n_embedding,
                "threshold": self.threshold,
                "user_id": filters["user_id"],
                "limit": limit,
            }
            ans = self.graph.query(cypher_query, params=params)
            result_relations.extend(ans)

        return result_relations

    def search(self, query, filters, limit=100):
        """
        Search for memories and related graph data.

        Args:
            query (str): Query to search for.
            filters (dict): A dictionary containing filters to be applied during the search.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.

        Returns:
            dict: A dictionary containing:
                - "contexts": List of search results from the base data store.
                - "entities": List of related graph data based on the query.
        """

        search_output = self._search(query, filters, limit)

        if not search_output:
            return []

        search_outputs_sequence = [[item["source"], item["relation"], item["destination"]] for item in search_output]
        bm25 = BM25Okapi(search_outputs_sequence)

        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=5)

        search_results = []
        for item in reranked_results:
            search_results.append({"source": item[0], "relationship": item[1], "target": item[2]})

        logger.info(f"Returned {len(search_results)} search results")

        return search_results

    def delete_all(self, filters):
        cypher = """
        MATCH (n {user_id: $user_id})
        DETACH DELETE n
        """
        params = {"user_id": filters["user_id"]}
        self.graph.query(cypher, params=params)

    def get_all(self, filters, limit=100):
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
        query = """
        MATCH (n {user_id: $user_id})-[r]->(m {user_id: $user_id})
        RETURN n.name AS source, type(r) AS relationship, m.name AS target
        LIMIT $limit
        """
        results = self.graph.query(query, params={"user_id": filters["user_id"], "limit": limit})

        final_results = []
        for result in results:
            final_results.append(
                {
                    "source": result["source"],
                    "relationship": result["relationship"],
                    "target": result["target"],
                }
            )

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results

    def _update_relationship(self, source, target, relationship, filters):
        """
        Update or create a relationship between two nodes in the graph.

        Args:
            source (str): The name of the source node.
            target (str): The name of the target node.
            relationship (str): The type of the relationship.
            filters (dict): A dictionary containing filters to be applied during the update.

        Raises:
            Exception: If the operation fails.
        """
        logger.info(f"Updating relationship: {source} -{relationship}-> {target}")

        relationship = relationship.lower().replace(" ", "_")

        # Check if nodes exist and create them if they don't
        check_and_create_query = """
        MERGE (n1 {name: $source, user_id: $user_id})
        MERGE (n2 {name: $target, user_id: $user_id})
        """
        self.graph.query(
            check_and_create_query,
            params={"source": source, "target": target, "user_id": filters["user_id"]},
        )

        # Delete any existing relationship between the nodes
        delete_query = """
        MATCH (n1 {name: $source, user_id: $user_id})-[r]->(n2 {name: $target, user_id: $user_id})
        DELETE r
        """
        self.graph.query(
            delete_query,
            params={"source": source, "target": target, "user_id": filters["user_id"]},
        )

        # Create the new relationship
        create_query = f"""
        MATCH (n1 {{name: $source, user_id: $user_id}}), (n2 {{name: $target, user_id: $user_id}})
        CREATE (n1)-[r:{relationship}]->(n2)
        RETURN n1, r, n2
        """
        result = self.graph.query(
            create_query,
            params={"source": source, "target": target, "user_id": filters["user_id"]},
        )

        if not result:
            raise Exception(f"Failed to update or create relationship between {source} and {target}")

if __name__ == "__main__":
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
            "provider": "openai",
            "config": {
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen-max",
                "temperature": 0.001,
                "top_p": 0.001,
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

    graph_mem = MemoryGraph(config)

    graph_mem.delete_all({"user_id": "wx"})

    data = """
    The name of Hancy's only cat is Kitty.
    The name of Hancy's only dog is Dylan.
    wx works for Giant Network Inc..
    """

    filter = {
        "user_id": "wx"
    }
    new_mem = graph_mem.add(data, filter)
    print(new_mem)

