"""Thin local Ollama wrapper for grounded generation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 512


class OllamaClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        try:
            from ollama import Client
        except ImportError as exc:
            raise RuntimeError("Install the 'ollama' Python package to use local generation.") from exc

        self.model = model
        self.host = host
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = Client(host=host)

    def generate(self, prompt: str, *, stream: bool = False) -> Iterator[str] | str:
        options = {"temperature": self.temperature, "num_predict": self.max_tokens}
        if stream:
            return self._stream(prompt, options)

        try:
            response = self._client.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                options=options,
            )
        except Exception as exc:
            raise RuntimeError(_generation_error(self.model, self.host)) from exc
        return _response_text(response)

    def health(self) -> bool:
        try:
            self._client.list()
        except Exception:
            return False
        return True

    def _stream(self, prompt: str, options: dict[str, float | int]) -> Iterator[str]:
        try:
            chunks = self._client.generate(
                model=self.model,
                prompt=prompt,
                stream=True,
                options=options,
            )
            for chunk in chunks:
                delta = _response_text(chunk)
                if delta:
                    yield delta
        except Exception as exc:
            raise RuntimeError(_generation_error(self.model, self.host)) from exc


def _response_text(response: Any) -> str:
    if isinstance(response, Mapping):
        value = response.get("response", "")
        return value if isinstance(value, str) else str(value)

    value = getattr(response, "response", "")
    return value if isinstance(value, str) else str(value)


def _generation_error(model: str, host: str) -> str:
    return (
        f"Ollama generation failed for model {model!r} at {host}. "
        f"Make sure Ollama is running and run `ollama pull {model}`."
    )
