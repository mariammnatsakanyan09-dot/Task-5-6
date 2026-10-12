import os
from google import genai

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Set GOOGLE_API_KEY first")
    exit(1)

client = genai.Client(api_key=api_key)
print(f"Listing models for key starting with {api_key[:10]}...\n")

try:
    models = client.models.list()
    for m in models:
        name = getattr(m, 'name', str(m))
        # name is like models/gemini-2.5-flash
        print(name)
except Exception as e:
    print(f"Error listing models: {e}")
    # fallback: try with different api versions via raw request
    import requests
    for ver in ["v1beta", "v1"]:
        url = f"https://generativelanguage.googleapis.com/{ver}/models?key={api_key}"
        r = requests.get(url)
        print(f"\n{ver} -> {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            for mdl in data.get('models', [])[:20]:
                print(f"  {mdl.get('name')} - methods: {mdl.get('supportedGenerationMethods')}")
        else:
            print(r.text[:500])
