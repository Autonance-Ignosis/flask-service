import os
import faiss
import json
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2") 

def get_chunks(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def embed_and_index(folder="/Users/krispatel/Desktop/Autonance/flask-service/chatbot/docs"):
    texts, metadata = [], []
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
            raw = f.read()
        chunks = get_chunks(raw)
        texts.extend(chunks)
        metadata.extend([{"source": file, "text": c} for c in chunks])

    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    with open("metadata.json", "w") as f:
        json.dump(metadata, f)
    faiss.write_index(index, "vector.index")

embed_and_index()
