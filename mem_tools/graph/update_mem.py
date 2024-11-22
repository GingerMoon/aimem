import logging
import os

from infra.llms.consts import *
from infra.llms.base import LLMBase
from infra.llms.factory import LlmFactory

async def update_mem(llm: LLMBase, existing_memories, new_memories):
    prompt = _prompt_dict.get(llm.config.model, _DEFAULT_PROMPT).format(existing_memories=existing_memories, memory=new_memories)
    tools = [_tool_description_update.get(llm.config.model, _DEFAULT_TOOL_DESC_UPDATE),
             _tool_description_add.get(llm.config.model, _DEFAULT_TOOL_DESC_ADD),
             _tool_description_noop.get(llm.config.model, _DEFAULT_TOOL_DESC_NOOP)]
    memory_updates = llm.generate_response(
        messages=[
            {
                ROLE: USER,
                CONTENT: prompt,
            },
        ],
        tools=tools,
    )
    logging.info(memory_updates)

    to_be_added = []
    to_be_updated = []
    for item in memory_updates[TOOL_CALLS]:
        if item[NAME] == "add_graph_memory":
            to_be_added.append(item[ARGUMENTS])
        elif item[NAME] == "update_graph_memory":
            to_be_updated.append(item[ARGUMENTS])
        elif item[NAME] == "noop":
            continue

    return to_be_added, to_be_updated


_prompt_dict = dict()
_tool_description_update = dict()
_tool_description_add = dict()
_tool_description_noop = dict()

_DEFAULT_PROMPT = """
You are an AI expert specializing in graph memory management and optimization. Your task is to analyze existing graph memories alongside new information, and update the relationships in the memory list to ensure the most accurate, current, and coherent representation of knowledge.

Input:
1. Existing Graph Memories: A list of current graph memories, each containing source, target, and relationship information.
2. New Graph Memory: Fresh information to be integrated into the existing graph structure.

Guidelines:
1. Identification: Use the source and target as primary identifiers when matching existing memories with new information.
2. Conflict Resolution:
   - If new information contradicts an existing memory:
     a) For matching source and target but differing content, update the relationship of the existing memory.
     b) If the new memory provides more recent or accurate information, update the existing memory accordingly.
3. Comprehensive Review: Thoroughly examine each existing graph memory against the new information, updating relationships as necessary. Multiple updates may be required.
4. Consistency: Maintain a uniform and clear style across all memories. Each entry should be concise yet comprehensive.
5. Semantic Coherence: Ensure that updates maintain or improve the overall semantic structure of the graph.
6. Temporal Awareness: If timestamps are available, consider the recency of information when making updates.
7. Relationship Refinement: Look for opportunities to refine relationship descriptions for greater precision or clarity.
8. Redundancy Elimination: Identify and merge any redundant or highly similar relationships that may result from the update.

Task Details:
- Existing Graph Memories:
{existing_memories}

- New Graph Memory: {memory}

Output:
Provide a list of update instructions, each specifying the source, target, and the new relationship to be set. Only include memories that require updates.
"""

SEARCH_FUNCTION = "search_"
SEARCH_PARAMETER_NODES = "nodes"
SEARCH_PARAMETER_RELATIONS = "relations"

_DEFAULT_TOOL_DESC_UPDATE = {
    "type": "function",
    "function": {
        "name": "update_graph_memory",
        "description": "Update the relationship key of an existing graph memory based on new information. This function should be called when there's a need to modify an existing relationship in the knowledge graph. The update should only be performed if the new information is more recent, more accurate, or provides additional context compared to the existing information. The source and destination nodes of the relationship must remain the same as in the existing graph memory; only the relationship itself can be updated.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The identifier of the source node in the relationship to be updated. This should match an existing node in the graph.",
                },
                "destination": {
                    "type": "string",
                    "description": "The identifier of the destination node in the relationship to be updated. This should match an existing node in the graph.",
                },
                "relationship": {
                    "type": "string",
                    "description": "The new or updated relationship between the source and destination nodes. This should be a concise, clear description of how the two nodes are connected.",
                },
            },
            "required": ["source", "destination", "relationship"],
            "additionalProperties": False,
        },
    },
}

_DEFAULT_TOOL_DESC_ADD = {
    "type": "function",
    "function": {
        "name": "add_graph_memory",
        "description": "Add a new graph memory to the knowledge graph. This function creates a new relationship between two nodes, potentially creating new nodes if they don't exist.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The identifier of the source node in the new relationship. This can be an existing node or a new node to be created.",
                },
                "destination": {
                    "type": "string",
                    "description": "The identifier of the destination node in the new relationship. This can be an existing node or a new node to be created.",
                },
                "relationship": {
                    "type": "string",
                    "description": "The type of relationship between the source and destination nodes. This should be a concise, clear description of how the two nodes are connected.",
                },
                "source_type": {
                    "type": "string",
                    "description": "The type or category of the source node. This helps in classifying and organizing nodes in the graph.",
                },
                "destination_type": {
                    "type": "string",
                    "description": "The type or category of the destination node. This helps in classifying and organizing nodes in the graph.",
                },
            },
            "required": [
                "source",
                "destination",
                "relationship",
                "source_type",
                "destination_type",
            ],
            "additionalProperties": False,
        },
    },
}

_DEFAULT_TOOL_DESC_NOOP = {
    "type": "function",
    "function": {
        "name": "noop",
        "description": "No operation should be performed to the graph entities. This function is called when the system determines that no changes or additions are necessary based on the current input or context. It serves as a placeholder action when no other actions are required, ensuring that the system can explicitly acknowledge situations where no modifications to the graph are needed.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}


# "The given text does not provide any information about Cattie's owner or their age. Could you please provide more details or context?"
if __name__ == "__main__":
    config = {
        "api_key": os.environ.get("DASHSCOPE_API_KEY"),
        "model": "qwen-max",
        "temperature": 0.001,
        "top_p": 0.001,
        "max_tokens": 1500,
    }
    llm = LlmFactory.create("aliyun", config)
    existing_memories = [
        {
            "destination_node": "Hancy",
            "destination_type": "person",
            "relation": "HAS_CHILD",
            "source_node": "XiangWang",
            "source_type": "person",
        },
        {
            "destination_node": "playing football",
            "destination_type": "activity",
            "relation": "LIKES",
            "source_node": "Hancy",
            "source_type": "person",
        }
    ]
    new_memories = [
        {
            "destination_node": "Hancy",
            "destination_type": "person",
            "relation": "HAS_CHILD",
            "source_node": "XiangWang",
            "source_type": "person",
        },
        {
            "destination_node": "cat",
            "destination_type": "pet",
            "relation": "owns",
            "source_node": "Hancy",
            "source_type": "person",
        },
    ]
    to_be_added, to_be_updated = update_mem(llm, existing_memories, new_memories)
    logging.info(f"{to_be_added=}")
    logging.info(f"{to_be_updated=}")

    new_memories = [
        {
            "destination_node": "Hancy",
            "destination_type": "person",
            "relation": "HAS_CHILD",
            "source_node": "XiangWang",
            "source_type": "person",
        },
        {
            "destination_node": "playing football",
            "destination_type": "activity",
            "relation": "DOES_NOT_LIKE",
            "source_node": "Hancy",
            "source_type": "person",
        }
    ]
    to_be_added, to_be_updated = update_mem(llm, existing_memories, new_memories)
    logger.info(f"{to_be_added=}")
    logger.info(f"{to_be_updated=}")