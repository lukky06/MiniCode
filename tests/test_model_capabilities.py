from minicode_harness.models import ModelCapabilities, resolve_model_capabilities


def test_provider_capabilities_are_conservative_and_provider_specific() -> None:
    anthropic = resolve_model_capabilities("anthropic", "claude")
    qwen = resolve_model_capabilities("qwen", "qwen-plus")
    deepseek = resolve_model_capabilities("deepseek", "deepseek-v4-flash")
    kimi = resolve_model_capabilities("kimi", "kimi-k3")
    ollama = resolve_model_capabilities("ollama", "local")

    assert anthropic.context_window == 200_000
    assert anthropic.supports_prompt_cache
    assert qwen.context_window == 131_072
    assert qwen.max_output_tokens == 8_192
    assert deepseek.context_window == 1_000_000
    assert deepseek.max_output_tokens == 8_192
    assert kimi.context_window == 1_000_000
    assert kimi.max_output_tokens == 131_072
    assert kimi.supports_prompt_cache
    assert ollama.context_window == 32_000
    assert not ollama.supports_prompt_cache


def test_reserved_output_never_exceeds_model_output_limit() -> None:
    capabilities = ModelCapabilities(context_window=8_000, max_output_tokens=2_000)

    assert capabilities.reserved_output_tokens == 2_000
