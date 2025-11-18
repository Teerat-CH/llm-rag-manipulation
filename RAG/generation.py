from llama_cpp import Llama
import os, sys
import contextlib
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

def generate_response(prompt, model="local"):

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
        user_query = input("Enter your prompt (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        response = generate_response(user_query, model="gemini")
        print("AI Response:", response)