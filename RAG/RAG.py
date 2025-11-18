from .index import Index
from .generation import generate_response

class RAG:
    def __init__(self):
        self.index = Index()

    def add_document(self, document: str):
        self.index.add_document(document)

    def remove_document(self, document: str):
        self.index.remove_document(document)
    
    def query(self, user_query: str, model="local", k=5):
        relevant_docs = self.index.search(user_query, k)
        context = "\n".join(relevant_docs)
        instructions = "You are a helpful assistent, use the provided context to answer user's question as concise and accurately as possible. Say that you do not know if the answer is not contained in the context."
        prompt = f"Instructions:\n{instructions}\n\nContext:\n{context}\n\nQuestion: {user_query}\nAnswer:"
        response = generate_response(prompt, model=model)
        return response
    
    def retrieve_documents(self, user_query: str, k=5):
        return self.index.search(user_query, k)
    
if __name__ == "__main__":
    rag_system = RAG()
    rag_system.index.reset_index()
    sample_documents = [
        "The capital of France is Paris.",
        "The largest planet in our solar system is Jupiter.",
        "The Great Wall of China is visible from space.",
        "The human body has 206 bones.",
        "The speed of light is approximately 299,792 kilometers per second.",
        "My name is John Doe"
    ]
    for doc in sample_documents:
        rag_system.add_document(doc)
    
    print(rag_system.query("what is my name", model="gemini"))

    rag_system.remove_document("My name is John Doe")

    print(rag_system.query("what is my name", model="gemini"))