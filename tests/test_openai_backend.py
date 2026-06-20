from routed_memory_experts.openai_backend import build_openai_expert_prompt


def test_build_openai_prompt_uses_no_think_and_exact_phrases():
    prompt = build_openai_expert_prompt(
        "kubernetes-expert",
        "Use helm template, then helm unittest.",
        "How do I prove chart output?",
        ["helm template", "helm unittest"],
    )
    assert prompt.startswith("/no_think")
    assert "kubernetes-expert" in prompt
    assert "helm template | helm unittest" in prompt
    assert "How do I prove chart output?" in prompt
