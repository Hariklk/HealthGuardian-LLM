# src/llm_stub.py
import os
import httpx
import asyncio

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

async def ask_openai(prompt, model="gpt-4o-mini", max_tokens=150):
    if not OPENAI_API_KEY:
        return "(No OPENAI_API_KEY set) " + prompt[:200]
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {"model": model, "messages":[{"role":"system","content":"You provide safe lifestyle tips. Never provide medical diagnosis."},{"role":"user","content":prompt}], "max_tokens": max_tokens}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, json=data, headers=headers)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]

# wrapper to pass into recommend() (works sync by running event loop)
def llm_client_sync(prompt):
    try:
        return asyncio.get_event_loop().run_until_complete(ask_openai(prompt))
    except RuntimeError:
        # new loop if none
        import asyncio
        return asyncio.run(ask_openai(prompt))
