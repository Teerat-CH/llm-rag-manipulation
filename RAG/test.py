from google import genai
from dotenv import load_dotenv
import os
import requests

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is the capital of France?",
)
print(response.text)