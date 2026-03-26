#!/usr/bin/env python3
"""Tests for AI Provider Factory."""

import pytest

from multi_llm_provider.base import AIAnalyzer, AIProvider
from multi_llm_provider.claude_api import ClaudeAPIAnalyzer
from multi_llm_provider.clipboard import ClipboardAnalyzer
from multi_llm_provider.factory import AIProviderFactory, ConfigError
from multi_llm_provider.mistral_api import MistralAPIAnalyzer
from multi_llm_provider.ollama import OllamaAnalyzer
from multi_llm_provider.openai_api import OpenAIAnalyzer


class TestAIProviderFactory:
    """Tests for AIProviderFactory."""

    def test_create_clipboard_analyzer(self):
        analyzer = AIProviderFactory.create("clipboard", {})
        assert isinstance(analyzer, ClipboardAnalyzer)
        assert isinstance(analyzer, AIAnalyzer)
        assert analyzer.provider == AIProvider.CLIPBOARD

    def test_create_clipboard_case_insensitive(self):
        analyzer1 = AIProviderFactory.create("CLIPBOARD", {})
        analyzer2 = AIProviderFactory.create("ClipBoard", {})
        analyzer3 = AIProviderFactory.create("clipboard", {})
        assert all(isinstance(a, ClipboardAnalyzer) for a in [analyzer1, analyzer2, analyzer3])

    def test_create_clipboard_whitespace_trimmed(self):
        analyzer = AIProviderFactory.create("  clipboard  ", {})
        assert isinstance(analyzer, ClipboardAnalyzer)

    def test_create_claude_analyzer_with_key(self):
        config = {
            "claude_api_key": "sk-ant-test-key-123",
            "claude_model": "claude-sonnet-4-20250514",
        }
        analyzer = AIProviderFactory.create("claude_api", config)
        assert isinstance(analyzer, ClaudeAPIAnalyzer)
        assert analyzer.provider == AIProvider.CLAUDE
        assert analyzer.model == "claude-sonnet-4-20250514"

    def test_create_claude_analyzer_missing_key(self):
        with pytest.raises(ConfigError) as excinfo:
            AIProviderFactory.create("claude_api", {})
        assert "Claude API key required" in str(excinfo.value)

    def test_create_claude_analyzer_default_model(self):
        config = {"claude_api_key": "sk-ant-test-key"}
        analyzer = AIProviderFactory.create("claude_api", config)
        assert analyzer.model == "claude-sonnet-4-20250514"

    def test_create_mistral_analyzer_with_key(self):
        config = {"mistral_api_key": "test-mistral-key", "mistral_model": "mistral-large-latest"}
        analyzer = AIProviderFactory.create("mistral_api", config)
        assert isinstance(analyzer, MistralAPIAnalyzer)
        assert analyzer.provider == AIProvider.MISTRAL
        assert analyzer.model == "mistral-large-latest"

    def test_create_mistral_analyzer_missing_key(self):
        with pytest.raises(ConfigError) as excinfo:
            AIProviderFactory.create("mistral_api", {})
        assert "Mistral API key required" in str(excinfo.value)

    def test_create_openai_analyzer_with_key(self):
        config = {"openai_api_key": "test-openai-key", "openai_model": "gpt-4-turbo"}
        analyzer = AIProviderFactory.create("openai", config)
        assert isinstance(analyzer, OpenAIAnalyzer)
        assert analyzer.provider == AIProvider.OPENAI
        assert analyzer.model == "gpt-4-turbo"

    def test_create_openai_analyzer_missing_key(self):
        with pytest.raises(ConfigError) as excinfo:
            AIProviderFactory.create("openai", {})
        assert "OpenAI API key required" in str(excinfo.value)

    def test_create_ollama_analyzer(self):
        config = {"ollama_host": "http://localhost:11434", "ollama_model": "mistral:7b"}
        analyzer = AIProviderFactory.create("ollama", config)
        assert isinstance(analyzer, OllamaAnalyzer)
        assert analyzer.provider == AIProvider.OLLAMA
        assert analyzer.model == "mistral:7b"
        assert analyzer.host == "http://localhost:11434"

    def test_create_ollama_analyzer_default_config(self):
        analyzer = AIProviderFactory.create("ollama", {})
        assert analyzer.model == "mistral:7b"
        assert analyzer.host == "http://localhost:11434"

    def test_create_invalid_provider_raises_error(self):
        with pytest.raises(ConfigError) as excinfo:
            AIProviderFactory.create("invalid_provider", {})
        assert "Unknown AI provider" in str(excinfo.value)
        assert "invalid_provider" in str(excinfo.value)

    def test_get_available_providers_returns_dict(self):
        providers = AIProviderFactory.get_available_providers()
        assert isinstance(providers, dict)
        assert len(providers) == 5

    def test_get_available_providers_has_all_providers(self):
        providers = AIProviderFactory.get_available_providers()
        expected = ["clipboard", "claude_api", "mistral_api", "openai", "ollama"]
        for provider in expected:
            assert provider in providers

    def test_get_available_providers_has_descriptions(self):
        providers = AIProviderFactory.get_available_providers()
        for _provider, description in providers.items():
            assert isinstance(description, str)
            assert len(description) > 0

    def test_validate_provider_config_clipboard_always_valid(self):
        is_valid, message = AIProviderFactory.validate_provider_config("clipboard", {})
        assert is_valid is True
        assert "ready" in message.lower() or "valid" in message.lower()

    def test_validate_provider_config_claude_with_key(self):
        config = {"claude_api_key": "sk-ant-test-key"}
        is_valid, message = AIProviderFactory.validate_provider_config("claude_api", config)
        assert is_valid is True
        assert "valid" in message.lower()

    def test_validate_provider_config_claude_without_key(self):
        is_valid, message = AIProviderFactory.validate_provider_config("claude_api", {})
        assert is_valid is False
        assert "missing" in message.lower()

    def test_validate_provider_config_mistral_with_key(self):
        config = {"mistral_api_key": "test-key"}
        is_valid, message = AIProviderFactory.validate_provider_config("mistral_api", config)
        assert is_valid is True

    def test_validate_provider_config_mistral_without_key(self):
        is_valid, message = AIProviderFactory.validate_provider_config("mistral_api", {})
        assert is_valid is False
        assert "MISTRAL_API_KEY" in message

    def test_validate_provider_config_openai_with_key(self):
        config = {"openai_api_key": "test-key"}
        is_valid, message = AIProviderFactory.validate_provider_config("openai", config)
        assert is_valid is True

    def test_validate_provider_config_openai_without_key(self):
        is_valid, message = AIProviderFactory.validate_provider_config("openai", {})
        assert is_valid is False
        assert "OPENAI_API_KEY" in message

    def test_validate_provider_config_ollama_always_valid(self):
        is_valid, message = AIProviderFactory.validate_provider_config("ollama", {})
        assert is_valid is True
        assert "valid" in message.lower()

    def test_validate_provider_config_unknown_provider(self):
        is_valid, message = AIProviderFactory.validate_provider_config("unknown", {})
        assert is_valid is False
        assert "unknown" in message.lower()
