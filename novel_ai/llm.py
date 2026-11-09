import json
import re
import httpx
from .config import settings
from .prompts import SYSTEM

class GenerationError(RuntimeError): pass

class LLM:
    def __init__(self, client=None):
        self.s = settings()
        self.client = client or httpx.Client(timeout=httpx.Timeout(self.s.model_timeout_seconds, connect=20))

    def generate(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 3000) -> str:
        url = self.s.model_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": "Bearer " + (self.s.model_api_key or "unused")}
        payload = {"model": self.s.model_name, "messages": [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}], "temperature": temperature,
            "max_tokens": max_tokens}
        try:
            response = self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip(): raise ValueError("Empty completion")
            return content.strip()
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise GenerationError("Model generation failed") from exc

    def structured(self, prompt: str, *, max_tokens: int = 3000) -> dict:
        raw = self.generate(prompt, temperature=0.25, max_tokens=max_tokens)
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I).strip()
        try:
            result = json.loads(clean)
        except json.JSONDecodeError as exc:
            raise GenerationError("Model returned invalid JSON; operation can be retried") from exc
        if not isinstance(result, dict): raise GenerationError("Expected JSON object")
        return result
