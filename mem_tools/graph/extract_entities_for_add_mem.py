import logging
import os

from infra.llms.consts import *
from infra.llms.base import LLMBase
from infra.llms.factory import LlmFactory

logger = logging.getLogger(__name__)

def extract_entities_for_add_mem(user_id: str, llm: LLMBase, data: str):
    prompt = _prompt_dict.get(llm.config.model, _DEFAULT_PROMPT).format(user_id=user_id)
    tools = [_tool_description.get(llm.config.model, _DEFAULT_TOOL_DESC)]
    extracted_entities = llm.generate_response(
        messages=[
            {
                ROLE: SYSTEM,
                CONTENT: prompt,
            },
            {ROLE: USER, CONTENT: data},
        ],
        tools=tools,
    )
    logger.info(extracted_entities)

    if extracted_entities[TOOL_CALLS]:
        extracted_entities = extracted_entities[TOOL_CALLS][0][ARGUMENTS][ENTITIES]
    else:
        extracted_entities = []
    return extracted_entities


SOURCE_NODE = "source_node"
SOURCE_TYPE = "source_type"
RELATION = "relation"
DESTINATION_NODE = "destination_node"
DESTINATION_TYPE = "destination_type"

_prompt_dict = dict()
_tool_description = dict()

_DEFAULT_PROMPT = """

You are an advanced algorithm designed to extract structured information from text to construct knowledge graphs. Your goal is to capture comprehensive information while maintaining accuracy. Follow these key principles:

1. Extract only explicitly stated information from the text.
2. Identify nodes (entities/concepts), their types, and relationships.
3. Use {user_id} as the source node for any self-references (I, me, my, etc.) in user messages.
CUSTOM_PROMPT

Nodes and Types:
- Aim for simplicity and clarity in node representation.
- Use basic, general types for node labels (e.g. "person" instead of "mathematician").

Relationships:
- Use consistent, general, and timeless relationship types.
- Example: Prefer "PROFESSOR" over "BECAME_PROFESSOR".

Entity Consistency:
- Use the most complete identifier for entities mentioned multiple times.
- Example: Always use "John Doe" instead of variations like "Joe" or pronouns.

Strive for a coherent, easily understandable knowledge graph by maintaining consistency in entity references and relationship types.

Adhere strictly to these guidelines to ensure high-quality knowledge graph extraction."""

_DEFAULT_TOOL_DESC = {
    "type": "function",
    "function": {
        "name": "add_query",
        "description": "Add new entities and relationships to the graph based on the provided query.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            f"{SOURCE_NODE}": {"type": "string"},
                            f"{SOURCE_TYPE}": {"type": "string"},
                            f"{RELATION}": {"type": "string"},
                            f"{DESTINATION_NODE}": {"type": "string"},
                            f"{DESTINATION_TYPE}": {"type": "string"},
                        },
                        "required": [
                            f"{SOURCE_NODE}",
                            f"{SOURCE_TYPE}",
                            f"{RELATION}",
                            f"{DESTINATION_NODE}",
                            f"{DESTINATION_TYPE}",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}


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

    # 0 = {dict: 5} {'destination_node': 'Hancy', 'destination_type': 'person', 'relation': 'HAS_CHILD', 'source_node': 'XiangWang', 'source_type': 'person'}
    # 1 = {dict: 5} {'destination_node': 'playing football', 'destination_type': 'activity', 'relation': 'LIKES', 'source_node': 'Hancy', 'source_type': 'person'}
    # 2 = {dict: 5} {'destination_node': 'Cattie', 'destination_type': 'pet', 'relation': 'OWNS', 'source_node': 'Hancy', 'source_type': 'person'}
    data = """
    My daughter's name is Hancy.
    Hancy likes playing football.
    Hancy has a cat named Cattie.
    """
    extracted_entities = extract_entities_for_add_mem("XiangWang", llm, data)
    print(extracted_entities)