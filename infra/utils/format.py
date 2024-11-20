
import re


def strip_markdown_json_flag(input_string):
    # Pattern for json code block
    pattern = r'```json((.|\n)*?)```'

    match = re.search(pattern, input_string)
    if match:
        # If match found, strip the markdown json tags and return
        json_content = match.group(1).strip()
        return json_content
    else:
        # If no match found, return original string
        return input_string
