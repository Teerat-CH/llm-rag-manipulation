import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RAG.RAG import RAG


if __name__ == "__main__":
    rag = RAG()

    documents = [
        "Python is a high-level programming language.",
        "The Eiffel Tower is located in Paris, France.",
        "The Great Wall of China is one of the Seven Wonders of the World."
    ]

    for doc in documents:
        rag.add_document(doc)

    prompt = "recommend me the best camera for beginner"

    while True:
        new_document = input("Enter a document to add (or 'exit' to quit): ")
        rag.add_document(new_document)
        print(rag.query(prompt, model="gemini"))
        rag.remove_document(new_document)
        print(rag.query(prompt, model="gemini"))