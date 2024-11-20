import logging
import os

from infra.llms.consts import *
from infra.llms.base import LLMBase
from infra.llms.factory import LlmFactory
from mem_tools.graph.extract_entities_for_add_mem import (
    extract_entities_for_add_mem,
    SOURCE_NODE,
    DESTINATION_NODE
)

logger = logging.getLogger(__name__)

def extract_entities_for_search(user_id: str, llm: LLMBase, query: str):
    extracted_entities = extract_entities_for_add_mem(user_id, llm, query)
    logger.info(extracted_entities)

    node_list = []
    for item in extracted_entities:
        node_list.append(item[SOURCE_NODE])
        node_list.append(item[DESTINATION_NODE])

    node_list = list(set(node_list))
    node_list = [node.lower().replace(" ", "_") for node in node_list]
    return node_list

def extract_entities_for_search_2(user_id: str, llm: LLMBase, query: str):
    prompt = _prompt_dict.get(llm.config.model, _DEFAULT_PROMPT).format(user_id=user_id)
    tools = [_tool_description.get(llm.config.model, _DEFAULT_TOOL_DESC)]
    extracted_entities = llm.generate_response(
        messages=[
            {
                ROLE: SYSTEM,
                CONTENT: prompt,
            },
            {ROLE: USER, CONTENT: query},
        ],
        tools=tools,
    )
    logger.info(extracted_entities)

    node_list = []
    for item in extracted_entities[TOOL_CALLS]:
        if item[NAME] == SEARCH_FUNCTION:
            try:
                node_list.extend(item[ARGUMENTS][SEARCH_PARAMETER_NODES])
            except Exception as e:
                logger.error(f"Error in search tool: {e}")
        else:
            logger.error(f"unsupported function {item[NAME]}")

    node_list = list(set(node_list))
    node_list = [node.lower().replace(" ", "_") for node in node_list]
    return node_list


_prompt_dict = dict()
_tool_description = dict()

_DEFAULT_PROMPT = """
You are a smart assistant who understands the entities, their types, and relations in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use {user_id} as the source node. Extract the entities. ***DO NOT*** answer the question itself if the given text is a question.
"""

SEARCH_FUNCTION = "search_"
SEARCH_PARAMETER_NODES = "nodes"
SEARCH_PARAMETER_RELATIONS = "relations"

_DEFAULT_TOOL_DESC = {
    "type": "function",
    "function": {
        "name": f"{SEARCH_FUNCTION}",
        "description": "Search for nodes and relations in the graph.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                f"{SEARCH_PARAMETER_NODES}": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of nodes to search for.",
                },
                f"{SEARCH_PARAMETER_RELATIONS}": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of relations to search for.",
                },
            },
            "required": ["nodes", "relations"],
            "additionalProperties": False,
        },
    },
}


# The given text is a question, and it contains the following entities:
#
# - Cattie (Type: Pet)
# - Cattie's owner (Type: Person, related to Cattie by an ownership relation)
#
# Since the question is asking for the age of "Cattie's owner," we need to identify XiangWang as the source node if the user is referring to themselves. However, there is no self-reference in the question, so we do not use XiangWang as the source node. The key entity here is "Cattie's owner" whose age is being inquired about.
#
if __name__ == "__main__":
    config = {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-max",
        "temperature": 0.001,
        "top_p": 0.001,
        "max_tokens": 1500,
    }
    llm = LlmFactory.create("openai", config)
    query = """
    My daughter's name is Hancy.
    Hancy likes playing football.
    Hancy has a cat named Cattie.
    """
    extracted_entities = extract_entities_for_search("XiangWang", llm, query)
    print(extracted_entities)