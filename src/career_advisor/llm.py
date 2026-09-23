"""Lớp gọi LLM dùng chung, có cache đĩa.

Provider và tên model đọc từ cấu hình (`LLM_PROVIDER`, `LLM_MODEL`), không gán cứng trong code.
Cache khoá theo (provider, model, system, prompt, json_output), nên chạy lại không tốn quota
và kết quả lặp lại được.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path

# temperature 0 để cùng prompt cho cùng câu trả lời nhiều nhất có thể.
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 4096
# Lỗi tạm thời (quá tải, giới hạn tần suất) thì chờ rồi thử lại: 5, 10, 20, 40, 80 giây.
MAX_RETRIES = 5
RETRY_BASE_SECONDS = 5.0
# Server chạy model cục bộ theo chuẩn OpenAI (mlx_lm.server, llama-server, LM Studio, Ollama).
DEFAULT_LOCAL_URL = "http://127.0.0.1:8080/v1"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
# Mặc định của Ollama nhỏ; prompt dài hơn sẽ bị cắt đầu mà không báo lỗi.
OLLAMA_NUM_CTX = 8192
LOCAL_TIMEOUT_SECONDS = 600

_THINK = re.compile(r"<think>.*?</think>", flags=re.S)


def strip_thinking(text: str) -> str:
    """Bỏ phần suy nghĩ `<think>…</think>` mà một số model (Qwen) in ra trước câu trả lời."""
    return _THINK.sub("", text).strip()


def extract_json(text: str) -> str:
    """Lấy khối JSON trong câu trả lời, kể cả khi model bọc trong ``` hoặc thêm chữ xung quanh."""
    text = strip_thinking(text)
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    end = max(text.rfind("}"), text.rfind("]"))
    return text[start : end + 1] if start >= 0 and end > start else text


class LLMClient(ABC):
    provider: str = ""

    def __init__(self, model: str, cache_dir: Path | None = None) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self.calls = 0  # số lượt gọi thật, không tính lượt lấy từ cache
        self._sleep = time.sleep

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, calls={self.calls})"

    def complete(self, prompt: str, *, system: str | None = None, json_output: bool = False) -> str:
        path = self._cache_path(prompt, system, json_output)
        if path is not None and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        text = self._call_with_retry(prompt, system, json_output)
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

    def _call_with_retry(self, prompt: str, system: str | None, json_output: bool) -> str:
        for attempt in range(MAX_RETRIES + 1):
            try:
                return self._call(prompt, system, json_output)
            except Exception as exc:
                if attempt == MAX_RETRIES or not self._is_transient(exc):
                    raise
                self._sleep(RETRY_BASE_SECONDS * 2**attempt)
        raise AssertionError("không tới được đây")

    def _is_transient(self, exc: Exception) -> bool:
        return False

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
            # Không dùng function calling. Tắt để SDK khỏi in cảnh báo AFC ở mỗi lượt gọi.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        resp = self._sdk.models.generate_content(model=self.model, contents=prompt, config=config)
        return resp.text or ""

    def _is_transient(self, exc: Exception) -> bool:
        from google.genai import errors

        if not isinstance(exc, errors.APIError) or exc.code not in (429, 500, 503):
            return False
        # Hết quota theo ngày thì thử lại chỉ tốn thêm lượt gọi.
        return not (exc.code == 429 and "PerDay" in str(exc))


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

    def _is_transient(self, exc: Exception) -> bool:
        import anthropic

        return isinstance(exc, anthropic.RateLimitError | anthropic.InternalServerError)


class OpenAICompatibleClient(LLMClient):
    """Model chạy cục bộ qua API chuẩn OpenAI. Không gửi dữ liệu ra ngoài máy."""

    provider = "local"

    def __init__(self, model: str, base_url: str = DEFAULT_LOCAL_URL, cache_dir: Path | None = None) -> None:
        super().__init__(model, cache_dir)
        self.base_url = base_url.rstrip("/")

    def _call(self, prompt: str, system: str | None, json_output: bool) -> str:
        if json_output:
            prompt += "\n\nChỉ trả về JSON hợp lệ, không thêm chữ nào khác."
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": MAX_OUTPUT_TOKENS,
            # Tắt chế độ suy nghĩ của Qwen: nhanh hơn nhiều, và các việc ở đây không cần.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = _post_json(f"{self.base_url}/chat/completions", body)
        return strip_thinking(data["choices"][0]["message"]["content"] or "")

    def _is_transient(self, exc: Exception) -> bool:
        return isinstance(exc, urllib.error.HTTPError) and exc.code >= 500


def _post_json(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=LOCAL_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read())


class OllamaClient(LLMClient):
    """Model chạy cục bộ bằng Ollama, qua API gốc `/api/chat`.

    Dùng API gốc thay vì lớp tương thích OpenAI để tắt chế độ suy nghĩ (`think`), ép JSON
    (`format`) và đặt độ dài ngữ cảnh (`num_ctx`).
    """

    provider = "ollama"

    def __init__(self, model: str, base_url: str = DEFAULT_OLLAMA_URL, cache_dir: Path | None = None) -> None:
        super().__init__(model, cache_dir)
        self.base_url = base_url.rstrip("/")

    def _call(self, prompt: str, system: str | None, json_output: bool) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_ctx": OLLAMA_NUM_CTX, "num_predict": MAX_OUTPUT_TOKENS},
        }
        if json_output:
            body["format"] = "json"
        data = _post_json(f"{self.base_url}/api/chat", body)
        return strip_thinking(data["message"]["content"] or "")

    def _is_transient(self, exc: Exception) -> bool:
        return isinstance(exc, urllib.error.HTTPError) and exc.code >= 500


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


_PROVIDERS = {
    "gemini": (GeminiClient, "GEMINI_API_KEY"),
    "claude": (ClaudeClient, "ANTHROPIC_API_KEY"),
    "local": (OpenAICompatibleClient, None),
    "ollama": (OllamaClient, None),
}


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
    if key_name is None:
        default_url = DEFAULT_OLLAMA_URL if provider == "ollama" else DEFAULT_LOCAL_URL
        return cls(model, settings.get("LLM_BASE_URL", default_url), cache_dir)
    api_key = settings.get(key_name, "")
    if not api_key:
        raise ValueError(f"Chưa đặt {key_name} trong .env")
    return cls(model, api_key, cache_dir)
