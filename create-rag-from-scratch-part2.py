import requests
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

corpus_of_documents = [
    "Take a leisurely walk in the park and enjoy the fresh air.",
    "Visit a local museum and discover something new.",
    "Attend a live music concert and feel the rhythm.",
    "Go for a hike and admire the natural scenery.",
    "Have a picnic with friends and share some laughs.",
    "Explore a new cuisine by dining at an ethnic restaurant.",
    "Take a yoga class and stretch your body and mind.",
    "Join a local sports league and enjoy some friendly competition.",
    "Attend a workshop or lecture on a topic you're interested in.",
    "Visit an amusement park and ride the roller coasters.",
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = model.encode(corpus_of_documents)

# 可以直接使用 print 函数将编码后的嵌入向量打印出来查看
print(doc_embeddings)

query = "What's the best outside activity?"
query_embedding = model.encode([query])

similarities = cosine_similarity(query_embedding, doc_embeddings)
print("\n-----------similartities------------\n")
print(similarities)

indexed = list(enumerate(similarities[0]))
sorted_index = sorted(indexed, key=lambda x: x[1], reverse=True)

recommended_documents = []

print("\n-----------formatted scores------------\n")

for value, score in sorted_index:
    formatted_score = "{:.2f}".format(score)
    print(f"{formatted_score} => {corpus_of_documents[value]}")

    # 只推荐相似度大于 0.3 的文档
    if score > 0.3:
        recommended_documents.append(corpus_of_documents[value])

recommended_activities = "\n".join(recommended_documents)
user_input = "I like to go to museum"
full_response = []

prompt = """
You are a bot that makes recommendations for activities. You answer in very short sentences and do not include extra information.
These are potential activities:
{recommended_activities}
The user's query is: {user_input}
Provide the user with 2 recommended activities based on their query.
"""

full_prompt = prompt.format(
    user_input=user_input, recommended_activities=recommended_activities
)

url = "http://localhost:11434/api/generate"
data = {
    "model": "llama3.1",
    "prompt": prompt.format(
        user_input=user_input, recommended_activities=recommended_activities
    ),
}
headers = {"Content-Type": "application/json"}
response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)

print(response)

try:
    count = 0
    for line in response.iter_lines():
        if line:
            decoded_line = json.loads(line.decode("utf-8"))

            full_response.append(decoded_line["response"])
finally:
    response.close()

print("\n-----------response------------\n")
print("".join(full_response))
