import time
import httpx
from config import Config
start = time.time()
client = httpx.Client()
response = client.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {Config.groq_api_key}"},
    json={
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": "say hello"}]
    }
)
print(f"Raw Groq ping from your machine: {time.time()-start:.2f}s")