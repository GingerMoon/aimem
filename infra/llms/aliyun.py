import json
import logging
import os
from typing import Dict, List, Optional

import dashscope

from infra.llms.base import LLMBase
from infra.llms.config.base_cfg import BaseLlmConfig

logger = logging.getLogger(__name__)

class DashscopeLlm(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "qwen-max"

        if not self.config.api_key:
            self.config.api_key = os.getenv("DASHSCOPE_API_KEY")

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if(response.status_code != 200):
            logger.error(f"status code: {response.status_code}, message: {response.message}, request id: {response.request_id}")
            return

        output = response.output

        if tools:
            processed_response = {
                "content": output.choices[0].message.content,
                "tool_calls": [],
            }

            if output.choices[0].message.tool_calls:
                for tool_call in output.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                        }
                    )

            return processed_response
        else:
            return output.choices[0].message.content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """
        Generate a response based on the given messages using OpenAI.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """

        params = {
            "api_key": self.config.api_key,
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "result_format": "message"
        }

        # dashscope doesn't support response_format. response_format is openai feature: Structured Outputs
        # if response_format:
        #     params["response_format"] = response_format
        if tools:  # TODO: Remove tools if no issues found with new memory addition logic
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = dashscope.Generation.call(**params)
        return self._parse_response(response, tools)


if __name__ == "__main__":
    config = {
        "api_key": os.environ.get("DASHSCOPE_API_KEY"),
        "model": "qwen-max",
        "temperature": 0.001,
        "top_p": 0.001,
        "max_tokens": 1500,
    }
    base_config = BaseLlmConfig(**config)
    llm = DashscopeLlm(base_config)
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '你是谁？'}
    ]
    print(llm.generate_response(messages))