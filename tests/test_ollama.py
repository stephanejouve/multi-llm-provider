#!/usr/bin/env python3
"""Tests for Ollama Local LLM Provider."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from multi_llm_provider.base import AIProvider, WorkflowError
from multi_llm_provider.ollama import OllamaAnalyzer


class TestOllamaAnalyzer:

    def test_init_with_defaults(self):
        analyzer = OllamaAnalyzer()
        assert analyzer.provider == AIProvider.OLLAMA
        assert analyzer.model == "mistral:7b"
        assert analyzer.host == "http://localhost:11434"
        assert analyzer.api_url == "http://localhost:11434/api/generate"

    def test_init_with_custom_model(self):
        analyzer = OllamaAnalyzer(model="llama3.1:70b")
        assert analyzer.model == "llama3.1:70b"

    def test_init_with_custom_host(self):
        analyzer = OllamaAnalyzer(host="http://192.168.1.100:11434")
        assert analyzer.host == "http://192.168.1.100:11434"

    @patch("multi_llm_provider.ollama.requests.post")
    def test_analyze_session_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Complete analysis."}
        mock_post.return_value = mock_response

        analyzer = OllamaAnalyzer()
        result = analyzer.analyze_session("Analyze this")
        assert "Complete analysis" in result
        mock_post.assert_called_once()

    @patch("multi_llm_provider.ollama.requests.post")
    def test_analyze_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Cannot connect")
        analyzer = OllamaAnalyzer()
        with pytest.raises(WorkflowError) as exc_info:
            analyzer.analyze_session("Test")
        assert "connect" in str(exc_info.value).lower() or "ollama" in str(exc_info.value).lower()

    @patch("multi_llm_provider.ollama.requests.post")
    def test_analyze_timeout_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Timeout")
        analyzer = OllamaAnalyzer()
        with pytest.raises(WorkflowError):
            analyzer.analyze_session("Test")

    @patch("multi_llm_provider.ollama.requests.get")
    def test_validate_config_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        analyzer = OllamaAnalyzer()
        assert analyzer.validate_config() is True

    @patch("multi_llm_provider.ollama.requests.get")
    def test_validate_config_server_offline(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("Cannot connect")
        analyzer = OllamaAnalyzer()
        assert analyzer.validate_config() is False

    @patch("multi_llm_provider.ollama.requests.post")
    def test_system_prompt_prepended(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Analysis."}
        mock_post.return_value = mock_response

        analyzer = OllamaAnalyzer()
        analyzer.analyze_session("User data", system_prompt="You are a coach.")
        call_kwargs = mock_post.call_args[1]
        sent_prompt = call_kwargs["json"]["prompt"]
        assert sent_prompt.startswith("You are a coach.")
        assert "User data" in sent_prompt

    @patch("multi_llm_provider.ollama.requests.post")
    def test_no_system_prompt_sends_raw(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Analysis."}
        mock_post.return_value = mock_response

        analyzer = OllamaAnalyzer()
        analyzer.analyze_session("Raw user prompt only")
        call_kwargs = mock_post.call_args[1]
        sent_prompt = call_kwargs["json"]["prompt"]
        assert sent_prompt == "Raw user prompt only"

    @patch("multi_llm_provider.ollama.requests.get")
    def test_get_provider_info(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        analyzer = OllamaAnalyzer(model="llama3.1:70b")
        info = analyzer.get_provider_info()
        assert info["provider"] == "ollama"
        assert info["model"] == "llama3.1:70b"
        assert info["requires_api_key"] is False
