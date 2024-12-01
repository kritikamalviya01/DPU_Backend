import os
from flask import jsonify

from sentence_transformers import SentenceTransformer

def compare(result , answers):
    # if not os.path.exists(f"tmp/{filename}.txt"):
    #     return {"error": "File not found", "status_code": 404}
    
    # with open(f"tmp/{filename}.txt", 'r') as text:
    #     text1 = text.read()
    # with open(f"tmp/pre-defined.txt", 'r') as text:
    #     text2 = text.read()

    sentences = [result, answers]

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    embeddings = model.encode(sentences)

    return (embeddings[0]@embeddings[1])
