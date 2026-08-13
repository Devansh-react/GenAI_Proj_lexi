"""Mistral chat-completions client with a bounded request timeout."""
from __future__ import annotations
import httpx
from app.config import Settings

class GenerationUnavailable(RuntimeError): pass

class MistralClient:
    def __init__(self, settings: Settings) -> None: self.settings = settings
    def generate(self, prompt: str) -> str:
        if not self.settings.mistral_api_key:
            raise GenerationUnavailable("Mistral API key is not configured")
        response = httpx.post("https://api.mistral.ai/v1/chat/completions", headers={"Authorization": f"Bearer {self.settings.mistral_api_key}"}, json={"model": self.settings.mistral_model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}, timeout=30.0)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
