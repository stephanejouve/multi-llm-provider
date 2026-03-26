#!/usr/bin/env python3
"""Tests for AI Providers base classes."""

import pytest

from multi_llm_provider.base import AIAnalyzer, AIProvider, WorkflowError


class TestAIProviderEnum:
    """Tests for AIProvider enum."""

    def test_enum_has_all_providers(self):
        """Test that enum contains all 5 expected providers."""
        expected_providers = [
            "clipboard",
            "claude_api",
            "mistral_api",
            "openai",
            "ollama",
        ]
        actual_providers = [p.value for p in AIProvider]
        assert len(actual_providers) == 5, f"Expected 5 providers, got {len(actual_providers)}"
        for provider in expected_providers:
            assert provider in actual_providers, f"Provider {provider} not in enum"

    def test_enum_values_are_strings(self):
        """Test that all enum values are strings."""
        for provider in AIProvider:
            assert isinstance(provider.value, str), f"Provider {provider.name} value is not string"

    def test_enum_clipboard_exists(self):
        assert AIProvider.CLIPBOARD.value == "clipboard"

    def test_enum_claude_exists(self):
        assert AIProvider.CLAUDE.value == "claude_api"

    def test_enum_mistral_exists(self):
        assert AIProvider.MISTRAL.value == "mistral_api"

    def test_enum_openai_exists(self):
        assert AIProvider.OPENAI.value == "openai"

    def test_enum_ollama_exists(self):
        assert AIProvider.OLLAMA.value == "ollama"

    def test_enum_from_string(self):
        provider = AIProvider("clipboard")
        assert provider == AIProvider.CLIPBOARD

    def test_enum_invalid_value_raises_error(self):
        with pytest.raises(ValueError):
            AIProvider("invalid_provider")


class TestWorkflowError:
    """Tests for WorkflowError."""

    def test_workflow_error_is_exception(self):
        assert issubclass(WorkflowError, Exception)

    def test_workflow_error_message(self):
        err = WorkflowError("test error")
        assert str(err) == "test error"


class TestAIAnalyzerABC:
    """Tests for AIAnalyzer abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError) as excinfo:
            AIAnalyzer()
        assert "Can't instantiate abstract class" in str(excinfo.value)

    def test_subclass_must_implement_analyze_session(self):
        class IncompleteAnalyzer(AIAnalyzer):
            pass

        with pytest.raises(TypeError) as excinfo:
            IncompleteAnalyzer()
        assert "analyze_session" in str(excinfo.value)

    def test_subclass_can_use_default_get_provider_info(self):
        class SimpleAnalyzer(AIAnalyzer):
            def __init__(self):
                super().__init__()
                self.provider = AIProvider.CLIPBOARD

            def analyze_session(self, prompt, dataset=None):
                return "test"

        analyzer = SimpleAnalyzer()
        info = analyzer.get_provider_info()
        assert isinstance(info, dict)
        assert info["provider"] == "clipboard"

    def test_subclass_can_use_default_validate_config(self):
        class SimpleAnalyzer(AIAnalyzer):
            def __init__(self):
                super().__init__()
                self.provider = AIProvider.CLIPBOARD

            def analyze_session(self, prompt, dataset=None):
                return "test"

        analyzer = SimpleAnalyzer()
        is_valid = analyzer.validate_config()
        assert is_valid is True

    def test_complete_subclass_can_be_instantiated(self):
        class CompleteAnalyzer(AIAnalyzer):
            def __init__(self):
                super().__init__()
                self.provider = AIProvider.CLIPBOARD

            def analyze_session(self, prompt, dataset=None):
                return "test analysis"

            def get_provider_info(self):
                return {"provider": "test"}

            def validate_config(self):
                return True

        analyzer = CompleteAnalyzer()
        assert analyzer is not None
        assert analyzer.provider == AIProvider.CLIPBOARD

    def test_subclass_analyze_session_signature(self):
        class TestAnalyzer(AIAnalyzer):
            def __init__(self):
                super().__init__()
                self.provider = AIProvider.CLIPBOARD

            def analyze_session(self, prompt, dataset=None):
                return f"Analyzed: {prompt[:20]}"

            def get_provider_info(self):
                return {}

            def validate_config(self):
                return True

        analyzer = TestAnalyzer()
        result = analyzer.analyze_session("Test prompt", dataset=None)
        assert result == "Analyzed: Test prompt"

    def test_subclass_has_provider_attribute(self):
        class TestAnalyzer(AIAnalyzer):
            def __init__(self):
                super().__init__()
                self.provider = AIProvider.OLLAMA

            def analyze_session(self, prompt, dataset=None):
                return "test"

            def get_provider_info(self):
                return {}

            def validate_config(self):
                return True

        analyzer = TestAnalyzer()
        assert hasattr(analyzer, "provider")
        assert analyzer.provider == AIProvider.OLLAMA
