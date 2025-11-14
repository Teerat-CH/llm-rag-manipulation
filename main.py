from RAG.generation import generate_response
from RAG.index import Index

if __name__ == "__main__":
    index = Index()
    index.reset_index()
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
    
    print("Chat with the RAG system. Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        response = generate_response(query, model="local")
        print(f"AI: {response}")