"""List the free OpenRouter models that can run this system (they must support tool calling).

    python list_openrouter_models.py            # free models with tools
    python list_openrouter_models.py --all      # every free model

The list changes often, so look at it on the day you run the homework. No key is needed for the list.
Free models: 20 requests per minute; 50 per day (1000 per day once you have bought 10 credits).
"""
import json
import ssl
import sys
import urllib.request


def main():
    show_all = "--all" in sys.argv
    ctx = None
    try:                     # python.org Python on macOS ships without root certificates; certifi is installed with openai/httpx
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=60, context=ctx) as r:
        data = json.load(r)["data"]
    rows = []
    for m in data:
        mid = m.get("id", "")
        pricing = m.get("pricing", {})
        is_free = mid.endswith(":free") or (str(pricing.get("prompt")) == "0" and str(pricing.get("completion")) == "0")
        params = set(m.get("supported_parameters") or [])
        if not is_free or (not show_all and "tools" not in params):
            continue
        rows.append((m.get("context_length") or 0, mid, "tools" in params,
                     "structured_outputs" in params or "response_format" in params))
    rows.sort(reverse=True)
    print(f"{'model id':58s} {'context':>9s}  tools  structured")
    for ctx, mid, tools, st in rows:
        print(f"{mid:58s} {ctx:9d}  {'yes' if tools else 'no':5s}  {'yes' if st else 'no'}")
    print(f"\n{len(rows)} models. Use one with:  --model <id>   (several ids separated by commas = fallbacks)")


if __name__ == "__main__":
    main()
