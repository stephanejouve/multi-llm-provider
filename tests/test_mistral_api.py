#!/usr/bin/env python3
"""Tests for Mistral API Provider."""

from unittest.mock import MagicMock, patch

import pytest

from multi_llm_provider.base import AIProvider, WorkflowError
from multi_llm_provider.mistral_api import MistralAPIAnalyzer


@pytest.fixture
def valid_api_key():
    return "mistral-test-key-12345"


@pytest.fixture
def mock_mistral_client():
    with patch("multi_llm_provider.mistral_api.Mistral") as mock_class:
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.complete = MagicMock()
        mock_class.return_value = mock_client
        yield mock_client


class TestMistralAPIAnalyzer:

    def test_init_success(self, valid_api_key):
        analyzer = MistralAPIAnalyzer(api_key=valid_api_key, model="mistral-large-latest")
        assert analyzer.provider == AIProvider.MISTRAL
        assert analyzer.model == "mistral-large-latest"
        assert analyzer.temperature == 0.7

    def test_init_missing_api_key(self):
        with pytest.raises(WorkflowError):
            MistralAPIAnalyzer(api_key=None)

    def test_init_empty_api_key(self):
        with pytest.raises(WorkflowError):
            MistralAPIAnalyzer(api_key="")

    def test_init_default_model(self, valid_api_key):
        analyzer = MistralAPIAnalyzer(api_key=valid_api_key)
        assert analyzer.model == "mistral-large-latest"

    def test_init_custom_parameters(self, valid_api_key):
        analyzer = MistralAPIAnalyzer(
            api_key=valid_api_key,
            model="mistral-medium-latest",
            temperature=0.5,
            max_tokens=8000,
            timeout=120,
        )
        assert analyzer.model == "mistral-medium-latest"
        assert analyzer.temperature == 0.5
        assert analyzer.max_tokens == 8000
        assert analyzer.timeout == 120

    def test_analyze_session_success(self, valid_api_key, mock_mistral_client):
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Complete analysis."
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_mistral_client.chat.complete.return_value = mock_response

        analyzer = MistralAPIAnalyzer(api_key=valid_api_key)
        result = analyzer.analyze_session("Analyze this")
        assert "Complete analysis" in result
        mock_mistral_client.chat.complete.assert_called_once()

    def test_analyze_error(self, valid_api_key, mock_mistral_client):
        mock_mistral_client.chat.complete.side_effect = Exception("API error")
        analyzer = MistralAPIAnalyzer(api_key=valid_api_key)
        with pytest.raises(WorkflowError):
            analyzer.analyze_session("Test")

    def test_get_provider_info(self, valid_api_key):
        analyzer = MistralAPIAnalyzer(api_key=valid_api_key, model="mistral-medium-latest")
        info = analyzer.get_provider_info()
        assert info["provider"] == "mistral_api"
        assert info["model"] == "mistral-medium-latest"
        assert info["requires_api_key"] is True

    def test_validate_config_success(self, valid_api_key):
        analyzer = MistralAPIAnalyzer(api_key=valid_api_key)
        assert analyzer.validate_config() is True

    def test_system_prompt_replaces_default(self, valid_api_key, mock_mistral_client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Result."
        mock_mistral_client.chat.complete.return_value = mock_response

        analyzer = MistralAPIAnalyzer(api_key=valid_api_key)
        analyzer.analyze_session("User prompt", system_prompt="Custom system.")
        call_kwargs = mock_mistral_client.chat.complete.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Custom system."

    def test_no_system_prompt_no_system_message(self, valid_api_key, mock_mistral_client):
        """Without system_prompt, no system message is sent (only user)."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Result."
        mock_mistral_client.chat.complete.return_value = mock_response

        analyzer = MistralAPIAnalyzer(api_key=valid_api_key)
        analyzer.analyze_session("User prompt")
        call_kwargs = mock_mistral_client.chat.complete.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_system_and_user_are_separate_messages(self, valid_api_key, mock_mistral_client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Result."
        mock_mistral_client.chat.complete.return_value = mock_response

        analyzer = MistralAPIAnalyzer(api_key=valid_api_key)
        analyzer.analyze_session("User data", system_prompt="System context")
        call_kwargs = mock_mistral_client.chat.complete.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "User data" not in messages[0]["content"]
