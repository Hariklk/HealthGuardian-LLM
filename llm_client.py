
import os
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_API_URL = os.getenv("LLM_API_URL", "")  

async def ask_llm(prompt: str, max_tokens=250):

    if OPENAI_API_KEY:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        data = {
            "model":"gpt-4o-mini",  
            "messages":[{"role":"system","content":"You provide general health education and safe lifestyle tips. Never give a diagnosis."},
                        {"role":"user","content": prompt}],
            "max_tokens": max_tokens
        }
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, headers=headers, json=data)
            r.raise_for_status()
            j = r.json()
            return j["choices"][0]["message"]["content"]
    elif LLM_API_URL:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(LLM_API_URL, json={"prompt":prompt, "max_tokens":max_tokens})
            r.raise_for_status()
            return r.json().get("text","")
    else:
        return "LLM not configured. Set OPENAI_API_KEY or LLM_API_URL."
