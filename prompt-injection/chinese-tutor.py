import sys
import requests
import json

SYSTEM_PROMPT = """
You're a helpful Chinese tutor.
You will be given an English sentence and your job is to explain how to say that sentence in Chinese.
After the explanation, break down the sentence into words with pinyin and English definitions.

{user_sentence}
"""


def chinese_tutor(user_sentence):
    full_response = []
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3.1",
        "prompt": SYSTEM_PROMPT.format(user_sentence=user_sentence),
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
    user_sentence = sys.argv[1]
    print(chinese_tutor(user_sentence))
