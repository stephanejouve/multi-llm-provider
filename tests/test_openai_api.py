#!/usr/bin/env python3
"""Tests for OpenAI API Provider."""

from unittest.mock import MagicMock, patch

import pytest

from multi_llm_provider.base import AIProvider, WorkflowError
from multi_llm_provider.openai_api import OpenAIAnalyzer


@pytest.fixture
def valid_api_key():
    return "sk-test-openai-key-12345"


@pytest.fixture
def mock_openai_client():
    with patch("multi_llm_provider.openai_api.OpenAI") as mock_class:
        mock_client = MagicMock()
        mock_class.return_value = mock_client
        yield mock_client


class TestOpenAIAnalyzer:

    def test_init_success(self, valid_api_key):
        analyzer = OpenAIAnalyzer(api_key=valid_api_key, model="gpt-4-turbo")
        assert analyzer.provider == AIProvider.OPENAI
        assert analyzer.model == "gpt-4-turbo"

    def test_init_missing_api_key(self):
        with pytest.raises(WorkflowError):
            OpenAIAnalyzer(api_key=None, model="gpt-4-turbo")

    def test_init_empty_api_key(self):
        with pytest.raises(WorkflowError):
            OpenAIAnalyzer(api_key="", model="gpt-4-turbo")

    def test_init_default_model(self, valid_api_key):
        analyzer = OpenAIAnalyzer(api_key=valid_api_key)
        assert analyzer.model == "gpt-4-turbo"

    def test_analyze_session_success(self, valid_api_key, mock_openai_client):
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Complete analysis."
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_openai_client.chat.completions.create.return_value = mock_response

        analyzer = OpenAIAnalyzer(api_key=valid_api_key)
        result = analyzer.analyze_session("Analyze this")
        assert "Complete analysis" in result

    def test_analyze_error(self, valid_api_key, mock_openai_client):
        mock_openai_client.chat.completions.create.side_effect = Exception("API error")
        analyzer = OpenAIAnalyzer(api_key=valid_api_key)
        with pytest.raises(WorkflowError):
            analyzer.analyze_session("Test")

    def test_get_provider_info(self, valid_api_key):
        analyzer = OpenAIAnalyzer(api_key=valid_api_key, model="gpt-3.5-turbo")
        info = analyzer.get_provider_info()
        assert info["provider"] == "openai"
        assert info["model"] == "gpt-3.5-turbo"
        assert info["requires_api_key"] is True

    def test_validate_config_success(self, valid_api_key):
        analyzer = OpenAIAnalyzer(api_key=valid_api_key)
        assert analyzer.validate_config() is True

    def test_system_prompt_adds_system_message(self, valid_api_key, mock_openai_client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Result."
        mock_openai_client.chat.completions.create.return_value = mock_response

        analyzer = OpenAIAnalyzer(api_key=valid_api_key)
        analyzer.analyze_session("User prompt", system_prompt="You are a coach.")
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are a coach."}
        assert messages[1] == {"role": "user", "content": "User prompt"}

    def test_no_system_prompt_only_user_message(self, valid_api_key, mock_openai_client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Result."
        mock_openai_client.chat.completions.create.return_value = mock_response

        analyzer = OpenAIAnalyzer(api_key=valid_api_key)
        analyzer.analyze_session("User prompt")
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
