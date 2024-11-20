import json
import logging
import os

from infra.llms.consts import *
from infra.llms.base import LLMBase
from infra.llms.factory import LlmFactory
from infra.utils.format import strip_markdown_json_flag

logger = logging.getLogger(__name__)

def llm_plan_update_mem(llm: LLMBase, existing_memories, new_memories):

    prompt = _prompt_dict.get(llm.config.model, _DEFAULT_PROMPT).format(
        OPERATION_TYPE=OPERATION_TYPE, OPERATION_PRE_VALUE = OPERATION_PRE_VALUE,
        existing_memories=existing_memories, new_memories=new_memories)

    messages = [
        {
            ROLE: USER,
            CONTENT: prompt,
        },
    ]
    logger.info(f"messages:\n{messages}")

    resp = llm.generate_response(
        messages=messages,
        response_format={"type": "json_object"},
    )
    logger.info(f"new_memories_with_actions:\n{resp}")

    resp = strip_markdown_json_flag(resp)
    new_memories_with_actions = json.loads(resp)

    return new_memories_with_actions


OPERATION_TYPE = "type"
OPERATION_PRE_VALUE = "value_pre"

_prompt_dict = dict()

_DEFAULT_PROMPT = """You are a smart memory manager which controls the memory of a system.
    You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.

    Based on the above four operations, the memory will change.

    Compare newly retrieved facts with the existing memory. For each new fact, decide whether to:
    - ADD: Add it to the memory as a new element
    - UPDATE: Update an existing memory element
    - DELETE: Delete an existing memory element
    - NONE: Make no change (if the fact is already present or irrelevant)

    There are specific guidelines to select which operation to perform:

    1. **Add**: If the retrieved facts contain new information not present in the memory, then you have to add it by generating a new ID in the id field.
        - **Example**:
            - Old Memory:
                [
                    {{
                        "id" : "0",
                        "text" : "User is a software engineer"
                    }}
                ]
            - Retrieved facts: ["Name is John"]
            - New Memory:
                {{
                    "memory" : [
                        {{
                            "id" : "0",
                            "text" : "User is a software engineer",
                            {OPERATION_TYPE} : "NONE"
                        }},
                        {{
                            "id" : "1",
                            "text" : "Name is John",
                            {OPERATION_TYPE} : "ADD"
                        }}
                    ]

                }}

    2. **Update**: If the retrieved facts contain information that is already present in the memory but the information is totally different, then you have to update it. 
        If the retrieved fact contains information that conveys the same thing as the elements present in the memory, then you have to keep the fact which has the most information. 
        Example (a) -- if the memory contains "User likes to play cricket" and the retrieved fact is "Loves to play cricket with friends", then update the memory with the retrieved facts.
        Example (b) -- if the memory contains "Likes cheese pizza" and the retrieved fact is "Loves cheese pizza", then you do not need to update it because they convey the same information.
        If the direction is to update the memory, then you have to update it.
        Please keep in mind while updating you have to keep the same ID.
        Please note to return the IDs in the output from the input IDs only and do not generate any new ID.
        - **Example**:
            - Old Memory:
                [
                    {{
                        "id" : "0",
                        "text" : "I really like cheese pizza"
                    }},
                    {{
                        "id" : "1",
                        "text" : "User is a software engineer"
                    }},
                    {{
                        "id" : "2",
                        "text" : "User likes to play cricket"
                    }}
                ]
            - Retrieved facts: ["Loves chicken pizza", "Loves to play cricket with friends"]
            - New Memory:
                {{
                "memory" : [
                        {{
                            "id" : "0",
                            "text" : "Loves cheese and chicken pizza",
                            {OPERATION_TYPE} : "UPDATE",
                            {OPERATION_PRE_VALUE} : : "I really like cheese pizza"
                        }},
                        {{
                            "id" : "1",
                            "text" : "User is a software engineer",
                            {OPERATION_TYPE} : "NONE"
                        }},
                        {{
                            "id" : "2",
                            "text" : "Loves to play cricket with friends",
                            {OPERATION_TYPE} : "UPDATE",
                            {OPERATION_PRE_VALUE} : : "User likes to play cricket"
                        }}
                    ]
                }}


    3. **Delete**: If the retrieved facts contain information that contradicts the information present in the memory, then you have to delete it. Or if the direction is to delete the memory, then you have to delete it.
        Please note to return the IDs in the output from the input IDs only and do not generate any new ID.
        - **Example**:
            - Old Memory:
                [
                    {{
                        "id" : "0",
                        "text" : "Name is John"
                    }},
                    {{
                        "id" : "1",
                        "text" : "Loves cheese pizza"
                    }}
                ]
            - Retrieved facts: ["Dislikes cheese pizza"]
            - New Memory:
                {{
                "memory" : [
                        {{
                            "id" : "0",
                            "text" : "Name is John",
                            {OPERATION_TYPE} : "NONE"
                        }},
                        {{
                            "id" : "1",
                            "text" : "Loves cheese pizza",
                            {OPERATION_TYPE} : "DELETE"
                        }}
                ]
                }}

    4. **No Change**: If the retrieved facts contain information that is already present in the memory, then you do not need to make any changes.
        - **Example**:
            - Old Memory:
                [
                    {{
                        "id" : "0",
                        "text" : "Name is John"
                    }},
                    {{
                        "id" : "1",
                        "text" : "Loves cheese pizza"
                    }}
                ]
            - Retrieved facts: ["Name is John"]
            - New Memory:
                {{
                "memory" : [
                        {{
                            "id" : "0",
                            "text" : "Name is John",
                            {OPERATION_TYPE} : "NONE"
                        }},
                        {{
                            "id" : "1",
                            "text" : "Loves cheese pizza",
                            {OPERATION_TYPE} : "NONE"
                        }}
                    ]
                }}

    Below is the current content of my memory which I have collected till now. You have to update it in the following format only:

    ``
    {existing_memories}
    ``

    The new retrieved facts are mentioned in the triple backticks. You have to analyze the new retrieved facts and determine whether these facts should be added, updated, or deleted in the memory.

    ```
    {new_memories}
    ```

    Follow the instruction mentioned below:
    - Do not return anything from the custom few shot prompts provided above.
    - If the current memory is empty, then you have to add the new retrieved facts to the memory.
    - You should return the updated memory in only JSON format as shown above. The memory key should be the same if no changes are made.
    - If there is an addition, generate a new key and add the new memory corresponding to it.
    - If there is a deletion, the memory key-value pair should be removed from the memory.
    - If there is an update, the ID key should remain the same and only the value needs to be updated. The previous value should be returned as value_pre.

    Do not return anything except the JSON format.
    """

if __name__ == "__main__":
    config = {
        "api_key": os.environ.get("DASHSCOPE_API_KEY"),
        "model": "qwen-max",
        "temperature": 0.001,
        "top_p": 0.001,
        "max_tokens": 1500,
    }
    llm = LlmFactory.create("aliyun", config)

    existing_memories = []
    new_memories = ["Daughter's name is Hancy"]
    new_memories_with_actions = llm_plan_update_mem(llm, existing_memories, new_memories)
    logger.info(f"{new_memories_with_actions=}")

    existing_memories = [
        {
            "id": "100",
            "text": "Have only one cat named Kitty.",
        },
        {
            "id": "2",
            "text": "Love playing football.",
        },
    ]
    new_memories = [
        {"facts": ["Have only one cat named Mimi."]},
        {"facts": ["Have only one dog named WangCai."]},
        {"facts": ["Don't like playing football."]},
    ]
    new_memories_with_actions = llm_plan_update_mem(llm, existing_memories, new_memories)
    logger.info(f"{new_memories_with_actions=}")