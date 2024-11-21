import json
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
from mem_tools.graph.extract_nodes_for_search import extract_nodes_for_search
from mem_tools.graph.update_mem import update_mem
from consts import *

logger = logging.getLogger(__name__)


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

    # TODO store metadata in nodes/relations
    def add(self, messages, metadata, filters):
        """
        Adds data to the graph.

        Args:
            messages (array): The classic openai API chat completion messages.
            metadata (dict): A dictionary containing metadata, such as user_name=xiang.
            filters (dict): A dictionary containing filters such as namespace=n1.
        """

        data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])

        # retrieve the search results
        search_output = self._search(data, metadata, filters)

        extracted_entities = extract_entities_for_add_mem(metadata[USER_NAME], self.llm, data)
        logger.info(f"Extracted entities: {extracted_entities}")

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

        logger.info(f"Added {len(to_be_added)} new memories to the graph")

        return returned_entities

    def _search(self, query, metadata, filters, limit=100):
        node_list = extract_nodes_for_search(metadata[f"{USER_NAME}"], self.llm, query)
        logger.info(f"Node list for search query : {node_list}")

        result_relations = []
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
            RETURN n.name AS source, elementId(n) AS source_id, type(r) AS relation, elementId(r) AS relation_id, m.name AS destination, elementId(m) AS destination_id, similarity
            UNION
            MATCH (n)
            WHERE n.embedding IS NOT NULL AND n.{NAMESPACE} = ${NAMESPACE}
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

        for node in node_list:
            n_embedding = self.embedding_model.embed(node)
            params = {
                "n_embedding": n_embedding,
                "threshold": self.threshold,
                f"{NAMESPACE}": filters[f"{NAMESPACE}"],
                "limit": limit,
            }
            # srd : source_relation_destination
            srds = self.graph.query(cypher_query, params=params)
            result_relations.extend(srds)
            for srd in srds:
                node_list_hop1.add(srd["destination"])

        logger.info(f"{node_list_hop1=}")
        for node in node_list_hop1:
            n_embedding = self.embedding_model.embed(node)
            params = {
                "n_embedding": n_embedding,
                "threshold": self.threshold,
                f"{NAMESPACE}": filters[f"{NAMESPACE}"],
                "limit": limit,
            }
            srds = self.graph.query(cypher_query, params=params)
            result_relations.extend(srds)

        return result_relations

    def search(self, query, metadata, filters, limit=100):
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

        search_output = self._search(query, metadata, filters, limit)

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

        logger.info(f"{search_results=}")

        return search_results

    def delete_all(self, filters):
        cypher = f"""
        MATCH (n {{{NAMESPACE}: ${NAMESPACE}}})
        DETACH DELETE n
        """
        params = {f"{NAMESPACE}": filters[f"{NAMESPACE}"]}
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

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results

    def _update_relationship(self, source, target, relationship, filters):
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
        logger.info(f"Updating relationship: {source} -{relationship}-> {target}")

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

    filters = {f"{NAMESPACE}": "n1"}
    mem.delete_all(filters)

    content = "My only daughter's name is Hancy. Hancy likes playing football. I work for Giant Network Inc.."
    messages = [{"role": "user", "content": content}]
    memories = mem.add(messages, metadata, filters)

    content = "Hancy doesn't likes playing football. Giant Network Inc. is located in SongJiang District Shanghai China."
    messages = [{"role": "user", "content": content}]
    memories = mem.add(messages, metadata, filters)
    logger.info(f"{memories=}")


