from llama_cpp import Llama
import os, sys
import contextlib
from .index import Index
from google import genai
from dotenv import load_dotenv
import os
import requests

load_dotenv()

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

index = Index()



model_path = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

with suppress_output():
    llm = Llama.from_pretrained(
        repo_id=model_path,
        filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_ctx=4096,
        verbose=False
    )

def generate_prompt(query):
    retrieved_docs = index.search(query, k=3)

    context_sections = "\n\n".join(
        [f"Context {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)]
    )

    prompt = (
        "You are a helpful AI assistant. Use the following contextual information "
        "to answer the user's question as accurately as possible.\n"
        "always refer to the context provided and tell users which context you get the information from.\n"
        "only give out an answer do not describe thought process or explain more than necessary.\n"
        "If the answer cannot be found in the context, say you don't know.\n\n"
        f"User question:\n{query}\n\n"
        f"{context_sections}\n\n"
        "Answer:"
    )

    return prompt

def generate_response(query, model="local"):
    prompt = generate_prompt(query)

    if model == "local":
        output = llm(prompt, max_tokens=256, temperature=0.6, top_p=0.8)
        output = output["choices"][0]["text"].strip()
    else:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        output = response.text
    return output

if __name__ == "__main__":
    while True:
        user_query = input("Enter your question (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        response = generate_response(user_query, model="gemeni")
        print("AI Response:", response)
