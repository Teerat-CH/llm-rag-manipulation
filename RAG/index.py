import os
from .encode import preprocess, encode, normalize
import pickle
import faiss

import numpy as np

class Index:
    def __init__(self):
        self.documents = []
        self.index = faiss.IndexFlatIP(768)
        self.set = set()

    def add_document(self, document: str):
        if document in self.set:
            return
        self.documents.append(document)
        self.set.add(document)

        document = preprocess(document)

        embedding = np.array([normalize(encode(document))]).astype('float32')
        self.index.add(embedding)

    def remove_document(self, document: str):
        if document not in self.set:
            print(f"Document not found: {document}")
            return

        self.documents.remove(document)
        self.set.remove(document)

        self.index = faiss.IndexFlatIP(768)

        for doc in self.documents:
            doc = preprocess(doc)
            embedding = np.array([normalize(encode(doc))]).astype('float32')
            self.index.add(embedding)

    def search(self, query, k=5):
        query_embedding = np.array([normalize(encode(preprocess(query)))]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        if len(self.documents) == 0:
            return []
        results = [self.documents[i] for i in indices[0]]
        return results
    
    def load_index(self, document_file_path="RAG/index_files/documents.pkl", index_file_path="RAG/index_files/index.bin"):
        if os.path.exists(document_file_path) and os.path.exists(index_file_path):
            self.documents = pickle.load(open(document_file_path, "rb"))
            self.index = faiss.read_index(index_file_path)
            self.set = set(self.documents)
        else:
            raise ImportError("document file path or index file path does not exist.")

    def save_index(self):
        faiss.write_index(self.index, "index.bin")
        pickle.dump(self.documents, open("documents.pkl", "wb"))

    def reset_index(self):
        self.documents = []
        self.index = faiss.IndexFlatIP(768)
        self.set = set()

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

    query = "what is my name"
    results = index.search(query, 2)

    assert len(results) == 2
    print("sentence shouldn't be in here")
    print(results)

    index.add_document("My name is John Doe.")
    results = index.search(query, 2)
    print("sentence should be in here")
    print(results)

    index.remove_document("My name is John Doe.")
    results = index.search(query, 2)
    print("sentence shouldn't be in here")
    print(results)