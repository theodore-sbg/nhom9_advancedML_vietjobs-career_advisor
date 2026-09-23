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
