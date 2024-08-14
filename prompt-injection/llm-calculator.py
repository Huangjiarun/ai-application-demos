import sys
import requests
import json

SYSTEM_PROMPT = """
You will be given a calculation to perform.
Write a Python function named calculation that returns the result of that calculation.
Output only the code of the calculation function and nothing else. Don’t use markdown.

{user_prompt}
"""


def llm_calculator(user_prompt):
    code = build_code(user_prompt)
    exec(code, globals())
    return globals().get("calculation")()


def build_code(user_prompt):
    full_response = []
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3.1",
        "prompt": SYSTEM_PROMPT.format(user_prompt=user_prompt),
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)

    try:
        for line in response.iter_lines():
            if line:
                decoded_line = json.loads(line.decode("utf-8"))
                full_response.append(decoded_line["response"])
    finally:
        response.close()

    return "".join(full_response)


if __name__ == "__main__":
    user_prompt = sys.argv[1]
    print(llm_calculator(user_prompt))
