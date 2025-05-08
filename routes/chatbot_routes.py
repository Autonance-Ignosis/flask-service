import os
import json
import faiss
import numpy as np
import requests
from flask import Blueprint, request, jsonify
from sentence_transformers import SentenceTransformer

chatbot_bp = Blueprint("chatbot", __name__)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("/Users/krispatel/Desktop/Autonance/flask-service/vector.index")

with open("metadata.json") as f:
    metadata = json.load(f)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

print(f"MISTRAL_API_KEY: {MISTRAL_API_KEY}")


def retrieve_docs(query, top_k=3):
    """Retrieve top-k relevant documents from FAISS index based on the query."""
    query_vec = embedder.encode([query]).astype("float32")
    D, I = index.search(query_vec, top_k)
    return [metadata[i]["text"] for i in I[0]]

def get_mistral_response(query, context):
    """Use Mistral API to generate a response based on the query and context."""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-medium",
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant providing relevant answers based on the context provided."
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {query}\nAnswer:"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }

    try:
        response = requests.post(MISTRAL_API_URL, json=payload, headers=headers)

        # # Log response status and body for debugging
        # print(f"Response Status Code: {response.status_code}")
        # print(f"Response Body: {response.text}")

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return "Sorry, I couldn't generate a response from Mistral AI."
    except Exception as e:
        return f"Error during Mistral API request: {str(e)}"


@chatbot_bp.route("/respond", methods=["POST"])
def chatbot_response():
    """Answers queries based on stored knowledge using Mistral AI."""
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "Missing required field: question"}), 400

    # Retrieve relevant documents from FAISS (query the index)
    context = "\n".join(retrieve_docs(question))

    # Use Mistral AI to generate a response based on the context and question
    answer = get_mistral_response(question, context)

    return jsonify({"response": answer}), 200
