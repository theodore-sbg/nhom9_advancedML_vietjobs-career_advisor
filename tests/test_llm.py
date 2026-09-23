import pytest

from career_advisor import config, llm


def test_load_env_parses_comments_quotes_and_blank_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# ghi chú\n\nLLM_PROVIDER=gemini\nLLM_MODEL=\"m-1\"  \nX='y'\n")

    assert config.load_env(env) == {"LLM_PROVIDER": "gemini", "LLM_MODEL": "m-1", "X": "y"}


def test_load_env_missing_file_returns_empty(tmp_path):
    assert config.load_env(tmp_path / "none.env") == {}


def test_second_identical_call_is_served_from_cache(tmp_path):
    client = llm.FakeLLMClient(lambda prompt: prompt.upper(), cache_dir=tmp_path)

    first = client.complete("xin chào")
    second = client.complete("xin chào")

    assert first == second == "XIN CHÀO"
    assert client.calls == 1


def test_different_prompt_or_system_is_a_new_call(tmp_path):
    client = llm.FakeLLMClient(lambda prompt: "ok", cache_dir=tmp_path)

    client.complete("a")
    client.complete("b")
    client.complete("a", system="khác")

    assert client.calls == 3


def test_cache_persists_across_client_instances(tmp_path):
    llm.FakeLLMClient(lambda p: "lần 1", cache_dir=tmp_path).complete("q")
    second = llm.FakeLLMClient(lambda p: "lần 2", cache_dir=tmp_path)

    assert second.complete("q") == "lần 1"
    assert second.calls == 0


def test_cache_is_keyed_by_model(tmp_path):
    llm.FakeLLMClient(lambda p: "m1", model="m1", cache_dir=tmp_path).complete("q")
    other = llm.FakeLLMClient(lambda p: "m2", model="m2", cache_dir=tmp_path)

    assert other.complete("q") == "m2"


def test_no_cache_dir_means_no_caching():
    client = llm.FakeLLMClient(lambda p: "ok")

    client.complete("q")
    client.complete("q")

    assert client.calls == 2


def test_make_client_picks_provider_from_settings(tmp_path):
    gemini = llm.make_client(
        {"LLM_PROVIDER": "gemini", "LLM_MODEL": "g", "GEMINI_API_KEY": "k"}, cache_dir=tmp_path
    )
    claude = llm.make_client(
        {"LLM_PROVIDER": "claude", "LLM_MODEL": "c", "ANTHROPIC_API_KEY": "k"}, cache_dir=tmp_path
    )

    assert isinstance(gemini, llm.GeminiClient) and gemini.model == "g"
    assert isinstance(claude, llm.ClaudeClient) and claude.model == "c"


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"LLM_PROVIDER": "openai", "LLM_MODEL": "x"}, "LLM_PROVIDER"),
        ({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, "LLM_MODEL"),
        ({"LLM_PROVIDER": "gemini", "LLM_MODEL": "g"}, "GEMINI_API_KEY"),
        ({"LLM_PROVIDER": "claude", "LLM_MODEL": "c"}, "ANTHROPIC_API_KEY"),
    ],
)
def test_make_client_rejects_bad_settings(settings, message):
    with pytest.raises(ValueError, match=message):
        llm.make_client(settings)


def test_repr_never_contains_api_key():
    client = llm.make_client({"LLM_PROVIDER": "gemini", "LLM_MODEL": "g", "GEMINI_API_KEY": "secret-123"})

    assert "secret-123" not in repr(client)
    assert "secret-123" not in str(vars(client).get("model"))


class _Flaky(llm.FakeLLMClient):
    """Lỗi tạm thời `failures` lần rồi mới trả lời."""

    def __init__(self, failures: int, **kwargs):
        super().__init__(lambda p: "ok", **kwargs)
        self.failures = failures
        self.sleeps: list[float] = []
        self._sleep = self.sleeps.append

    def _call(self, prompt, system, json_output):
        if self.failures:
            self.failures -= 1
            raise TimeoutError("quá tải")
        return super()._call(prompt, system, json_output)

    def _is_transient(self, exc):
        return isinstance(exc, TimeoutError)


def test_transient_errors_are_retried_with_growing_waits():
    client = _Flaky(failures=2)

    assert client.complete("q") == "ok"
    assert client.sleeps == [llm.RETRY_BASE_SECONDS, llm.RETRY_BASE_SECONDS * 2]


def test_gives_up_after_max_retries():
    client = _Flaky(failures=llm.MAX_RETRIES + 1)

    with pytest.raises(TimeoutError):
        client.complete("q")


def test_non_transient_errors_are_not_retried():
    class Broken(llm.FakeLLMClient):
        def _call(self, prompt, system, json_output):
            raise ValueError("sai prompt")

    client = Broken(lambda p: "ok")
    client._sleep = lambda s: pytest.fail("không được chờ để thử lại")

    with pytest.raises(ValueError):
        client.complete("q")


def test_make_client_local_needs_no_key_and_has_default_url():
    client = llm.make_client({"LLM_PROVIDER": "local", "LLM_MODEL": "qwen"})

    assert isinstance(client, llm.OpenAICompatibleClient)
    assert client.base_url == llm.DEFAULT_LOCAL_URL


def test_strip_thinking_removes_think_block():
    assert llm.strip_thinking("<think>\nnghĩ lâu\n</think>\n\nHà Nội") == "Hà Nội"
    assert llm.strip_thinking("Hà Nội") == "Hà Nội"


def _serve(reply: str, captured: list):
    import http.server
    import json as _json
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            captured.append((self.path, _json.loads(body)))
            payload = _json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_local_client_posts_chat_completion_and_disables_thinking():
    captured: list = []
    server = _serve("<think>…</think>Hà Nội", captured)
    try:
        client = llm.OpenAICompatibleClient("qwen", f"http://127.0.0.1:{server.server_port}/v1")
        answer = client.complete("Thủ đô?", system="Trả lời ngắn")
    finally:
        server.shutdown()

    path, body = captured[0]
    assert answer == "Hà Nội"
    assert path == "/v1/chat/completions"
    assert body["model"] == "qwen" and body["temperature"] == 0
    assert body["messages"] == [
        {"role": "system", "content": "Trả lời ngắn"},
        {"role": "user", "content": "Thủ đô?"},
    ]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_gemini_daily_quota_is_not_retried_but_overload_is():
    from google.genai import errors

    client = llm.GeminiClient("g", "k")
    daily = errors.ClientError(
        429, {"error": {"message": "Quota exceeded ... PerDayPerProjectPerModel-FreeTier"}}
    )
    per_minute = errors.ClientError(429, {"error": {"message": "Quota exceeded ... PerMinute"}})
    overload = errors.ServerError(503, {"error": {"message": "high demand"}})

    assert not client._is_transient(daily)
    assert client._is_transient(per_minute)
    assert client._is_transient(overload)


def _serve_ollama(reply: str, captured: list):
    import http.server
    import json as _json
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            captured.append((self.path, _json.loads(body)))
            payload = _json.dumps({"message": {"role": "assistant", "content": reply}, "done": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_make_client_ollama_uses_default_url():
    client = llm.make_client({"LLM_PROVIDER": "ollama", "LLM_MODEL": "qwen3.5:9b"})

    assert isinstance(client, llm.OllamaClient)
    assert client.base_url == llm.DEFAULT_OLLAMA_URL


def test_ollama_client_disables_thinking_forces_json_and_sets_context():
    captured: list = []
    server = _serve_ollama('{"answers": []}', captured)
    try:
        client = llm.OllamaClient("qwen3.5:9b", f"http://127.0.0.1:{server.server_port}")
        answer = client.complete("Chấm các cặp", system="Chỉ trả JSON", json_output=True)
    finally:
        server.shutdown()

    path, body = captured[0]
    assert answer == '{"answers": []}'
    assert path == "/api/chat"
    assert body["think"] is False and body["stream"] is False
    assert body["format"] == "json"
    assert body["options"]["temperature"] == 0
    assert body["options"]["num_ctx"] == llm.OLLAMA_NUM_CTX
    assert body["messages"][0] == {"role": "system", "content": "Chỉ trả JSON"}


def test_ollama_client_plain_text_has_no_format():
    captured: list = []
    server = _serve_ollama("<think>x</think>Hà Nội", captured)
    try:
        answer = llm.OllamaClient("m", f"http://127.0.0.1:{server.server_port}").complete("Thủ đô?")
    finally:
        server.shutdown()

    assert answer == "Hà Nội"
    assert "format" not in captured[0][1]
