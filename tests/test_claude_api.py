#!/usr/bin/env python3
"""Tests for Claude API Provider."""

from unittest.mock import MagicMock, patch

import pytest

from multi_llm_provider.base import AIProvider, WorkflowError
from multi_llm_provider.claude_api import ClaudeAPIAnalyzer


@pytest.fixture
def valid_api_key():
    return "sk-ant-test-key-123"


@pytest.fixture
def mock_anthropic():
    with patch("multi_llm_provider.claude_api.Anthropic") as mock_class:
        mock_client = MagicMock()
        mock_class.return_value = mock_client
        yield mock_client


class TestClaudeAPIAnalyzer:

    def test_init_success(self, valid_api_key):
        analyzer = ClaudeAPIAnalyzer(api_key=valid_api_key, model="claude-sonnet-4-20250514")
        assert analyzer.provider == AIProvider.CLAUDE
        assert analyzer.model == "claude-sonnet-4-20250514"
        assert analyzer.client is not None

    def test_init_missing_api_key(self):
        with pytest.raises(WorkflowError) as exc_info:
            ClaudeAPIAnalyzer(api_key=None, model="claude-sonnet-4")
        error_msg = str(exc_info.value).lower()
        assert "api key" in error_msg or "format" in error_msg

    def test_init_invalid_api_key_format(self):
        with pytest.raises(WorkflowError) as exc_info:
            ClaudeAPIAnalyzer(api_key="invalid-key", model="claude-sonnet-4")
        error_msg = str(exc_info.value).lower()
        assert "sk-ant-" in error_msg or "format" in error_msg

    def test_init_default_model(self, valid_api_key):
        analyzer = ClaudeAPIAnalyzer(api_key=valid_api_key)
        assert analyzer.model == "claude-sonnet-4-20250514"

    def test_init_custom_max_tokens(self, valid_api_key):
        analyzer = ClaudeAPIAnalyzer(
            api_key=valid_api_key, model="claude-sonnet-4", max_tokens=8000
        )
        assert analyzer.max_tokens == 8000

    def test_analyze_session_success(self, valid_api_key, mock_anthropic):
        mock_message = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Complete session analysis."
        mock_message.content = [mock_content]
        mock_anthropic.messages.create.return_value = mock_message

        analyzer = ClaudeAPIAnalyzer(api_key=valid_api_key)
        result = analyzer.analyze_session("Analyze this session")
        assert "Complete session analysis" in result
        mock_anthropic.messages.create.assert_called_once()

    def test_analyze_with_dataset(self, valid_api_key, mock_anthropic):
        mock_message = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Analysis with data."
        mock_message.content = [mock_content]
        mock_anthropic.messages.create.return_value = mock_message

        analyzer = ClaudeAPIAnalyzer(api_key=valid_api_key)
        result = analyzer.analyze_session("Prompt", {"tss": 65})
        assert "Analysis with data" in result

    def test_analyze_empty_prompt(self, valid_api_key, mock_anthropic):
        mock_message = MagicMock()
        mock_content = MagicMock()
        mock_content.text = ""
        mock_message.content = [mock_content]
        mock_anthropic.messages.create.return_value = mock_message

        analyzer = ClaudeAPIAnalyzer(api_key=valid_api_key)
        result = analyzer.analyze_session("")
        assert result == ""

    def test_analyze_authentication_error(self, valid_api_key, mock_anthropic):
        mock_anthropic.messages.create.side_effect = Exception("Invalid API key")
        analyzer = ClaudeAPIAnalyzer(api_key=valid_api_key)
        with pytest.raises(Exception) as exc_info:
            analyzer.analyze_session("Test")
        assert "API key" in str(exc_info.value) or "Invalid" in str(exc_info.value)

    def test_get_provider_info(self, valid_api_key):
        analyzer = ClaudeAPIAnalyzer(api_key=valid_api_key, model="claude-3-opus")
        info = analyzer.get_provider_info()
        assert info["provider"] == "claude_api"
        assert info["model"] == "claude-3-opus"
        assert info["requires_api_key"] is True

    def test_validate_config_success(self, valid_api_key):
        analyzer = ClaudeAPIAnalyzer(api_key=valid_api_key)
        assert analyzer.validate_config() is True

    def test_system_prompt_passed_to_api(self, valid_api_key, mock_anthropic):
        mock_message = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Analysis with system prompt."
        mock_message.content = [mock_content]
        mock_anthropic.messages.create.return_value = mock_message

        analyzer = ClaudeAPIAnalyzer(api_key=valid_api_key)
        analyzer.analyze_session("User prompt", system_prompt="You are a coach.")
        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a coach."

    def test_no_system_prompt_omits_system_param(self, valid_api_key, mock_anthropic):
        mock_message = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Analysis."
        mock_message.content = [mock_content]
        mock_anthropic.messages.create.return_value = mock_message

        analyzer = ClaudeAPIAnalyzer(api_key=valid_api_key)
        analyzer.analyze_session("User prompt")
        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_system_prompt_not_in_user_messages(self, valid_api_key, mock_anthropic):
        mock_message = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "OK"
        mock_message.content = [mock_content]
        mock_anthropic.messages.create.return_value = mock_message

        system_text = "UNIQUE_SYSTEM_MARKER_XYZ"
        analyzer = ClaudeAPIAnalyzer(api_key=valid_api_key)
        analyzer.analyze_session("User data", system_prompt=system_text)
        call_kwargs = mock_anthropic.messages.create.call_args[1]
        for msg in call_kwargs["messages"]:
            if msg["role"] == "user":
                assert system_text not in msg["content"]
