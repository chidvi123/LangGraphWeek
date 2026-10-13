
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

url = "https://api.groq.com/openai/v1/models"

response = requests.get(
    url,
    headers={"Authorization": f"Bearer {api_key}"},
    timeout=30,
)

response.raise_for_status()
models = response.json()["data"]

print(f"\nAvailable models: {len(models)}\n")

for model in sorted(models, key=lambda m: m["id"]):
    print(model["id"])
