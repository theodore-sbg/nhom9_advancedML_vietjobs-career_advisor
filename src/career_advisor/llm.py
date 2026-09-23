"""Lớp gọi LLM dùng chung, có cache đĩa.

Provider và tên model đọc từ cấu hình (`LLM_PROVIDER`, `LLM_MODEL`), không gán cứng trong code.
Cache khoá theo (provider, model, system, prompt, json_output), nên chạy lại không tốn quota
và kết quả lặp lại được.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path

# temperature 0 để cùng prompt cho cùng câu trả lời nhiều nhất có thể.
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 4096


class LLMClient(ABC):
    provider: str = ""

    def __init__(self, model: str, cache_dir: Path | None = None) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self.calls = 0  # số lượt gọi thật, không tính lượt lấy từ cache

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, calls={self.calls})"

    def complete(self, prompt: str, *, system: str | None = None, json_output: bool = False) -> str:
        path = self._cache_path(prompt, system, json_output)
        if path is not None and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        text = self._call(prompt, system, json_output)
        self.calls += 1
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
        return text

    def _cache_path(self, prompt: str, system: str | None, json_output: bool) -> Path | None:
        if self.cache_dir is None:
            return None
        key = json.dumps([self.provider, self.model, system, prompt, json_output], ensure_ascii=False)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    @abstractmethod
    def _call(self, prompt: str, system: str | None, json_output: bool) -> str: ...


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, model: str, api_key: str, cache_dir: Path | None = None) -> None:
        super().__init__(model, cache_dir)
        self._api_key = api_key
        self._sdk = None  # tạo khi gọi lần đầu, để test không cần mạng

    def _call(self, prompt: str, system: str | None, json_output: bool) -> str:
        from google import genai
        from google.genai import types

        if self._sdk is None:
            self._sdk = genai.Client(api_key=self._api_key)
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="application/json" if json_output else None,
        )
        resp = self._sdk.models.generate_content(model=self.model, contents=prompt, config=config)
        return resp.text or ""


class ClaudeClient(LLMClient):
    provider = "claude"

    def __init__(self, model: str, api_key: str, cache_dir: Path | None = None) -> None:
        super().__init__(model, cache_dir)
        self._api_key = api_key
        self._sdk = None

    def _call(self, prompt: str, system: str | None, json_output: bool) -> str:
        import anthropic

        if self._sdk is None:
            self._sdk = anthropic.Anthropic(api_key=self._api_key)
        if json_output:
            prompt += "\n\nChỉ trả về JSON hợp lệ, không thêm chữ nào khác."
        kwargs = {"system": system} if system else {}
        resp = self._sdk.messages.create(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class FakeLLMClient(LLMClient):
    """LLM giả cho test: trả lời bằng hàm `respond(prompt)`."""

    provider = "fake"

    def __init__(
        self, respond: Callable[[str], str], model: str = "fake", cache_dir: Path | None = None
    ) -> None:
        super().__init__(model, cache_dir)
        self.respond = respond
        self.prompts: list[str] = []

    def _call(self, prompt: str, system: str | None, json_output: bool) -> str:
        self.prompts.append(prompt)
        return self.respond(prompt)


_PROVIDERS = {"gemini": (GeminiClient, "GEMINI_API_KEY"), "claude": (ClaudeClient, "ANTHROPIC_API_KEY")}


def make_client(settings: Mapping[str, str] | None = None, cache_dir: Path | None = None) -> LLMClient:
    """Tạo client theo `LLM_PROVIDER` và `LLM_MODEL`. Mặc định đọc từ `.env` và dùng cache chung."""
    from career_advisor import config

    if settings is None:
        settings = config.settings()
        cache_dir = cache_dir or config.CACHE_DIR / "llm"
    provider = settings.get("LLM_PROVIDER", "").lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"LLM_PROVIDER phải là một trong {sorted(_PROVIDERS)}, nhận {provider!r}")
    model = settings.get("LLM_MODEL", "")
    if not model:
        raise ValueError("Chưa đặt LLM_MODEL trong .env")
    cls, key_name = _PROVIDERS[provider]
    api_key = settings.get(key_name, "")
    if not api_key:
        raise ValueError(f"Chưa đặt {key_name} trong .env")
    return cls(model, api_key, cache_dir)
