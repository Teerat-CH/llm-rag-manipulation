import os
from .encode import preprocess, encode, normalize
import pickle
import faiss

import numpy as np

original_dir = os.getcwd()
os.chdir("RAG/index_files")

class Index:
    def __init__(self):
        if os.path.exists("index.bin"):
            self.documents = pickle.load(open("documents.pkl", "rb"))
            self.index = faiss.read_index("index.bin")
            self.set = set(self.documents)
        else:
            self.documents = []
            self.index = faiss.IndexFlatIP(768)
            self.set = set()

    def add_document(self, document: str):
        document = preprocess(document)
        if document in self.set:
            return
        self.documents.append(document)
        embedding = np.array([normalize(encode(document))]).astype('float32')
        self.index.add(embedding)

    def search(self, query, k=5):
        query_embedding = np.array([normalize(encode(preprocess(query)))]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        if len(self.documents) == 0:
            return []
        results = [self.documents[i] for i in indices[0]]
        return results

    def save_index(self):
        faiss.write_index(self.index, "index.bin")
        pickle.dump(self.documents, open("documents.pkl", "wb"))

    def reset_index(self):
        self.documents = []
        self.index = faiss.IndexFlatIP(768)
        self.set = set()
        if os.path.exists("index.bin"):
            os.remove("index.bin")
        if os.path.exists("documents.pkl"):
            os.remove("documents.pkl")

if __name__ == "__main__":
    index = Index()
    sample_documents = [
        "The capital of France is Paris.",
        "The largest planet in our solar system is Jupiter.",
        "The Great Wall of China is visible from space.",
        "The human body has 206 bones.",
        "The speed of light is approximately 299,792 kilometers per second."
    ]
    for doc in sample_documents:
        index.add_document(doc)
    index.save_index()
    query = "what is the capital of France?"
    results = index.search(query, 2)
    print("Search results:")
    for res in results:
        print(res)