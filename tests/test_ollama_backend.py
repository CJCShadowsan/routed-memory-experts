from routed_memory_experts.ollama_backend import build_expert_prompt


def test_build_expert_prompt_includes_specialist_context_and_required_phrases():
    prompt = build_expert_prompt(
        "python-expert",
        "Use pytest -q and assert the failure first.",
        "How should I test this?",
        ["pytest", "failure"],
    )
    assert "python-expert" in prompt
    assert "Use pytest -q" in prompt
    assert "pytest | failure" in prompt
    assert "How should I test this?" in prompt
