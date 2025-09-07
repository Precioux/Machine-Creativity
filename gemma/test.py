import requests

OLLAMA_BASE = "http://localhost:11434"   # change if your Ollama is bound elsewhere
MODEL = "gemma3:latest"                  # replace with your tag, e.g. "gemma3:2b"

try:
    r = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": MODEL, "prompt": "Say 'ready' if you can hear me.", "stream": False},
        timeout=30,
    )
    r.raise_for_status()
    resp = r.json()
    print("✅ Connected!")
    print("Model:", resp.get("model"))
    print("Response:", resp.get("response"))
except Exception as e:
    print("❌ Could not reach Gemma:", e)
