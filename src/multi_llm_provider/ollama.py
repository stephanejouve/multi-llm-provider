"""Ollama local LLM integration.

Local LLM provider via Ollama server.
100% free, privacy-first, no rate limits.
"""

import logging

import requests

from .base import AIAnalyzer, AIProvider, WorkflowError

logger = logging.getLogger(__name__)


class OllamaAnalyzer(AIAnalyzer):
    """Ollama local LLM provider.

    Run LLMs locally via Ollama server. Free, private, unlimited.

    Attributes:
        provider: AIProvider.OLLAMA
        model: Ollama model name (llama3.1, mistral, etc.)
        host: Ollama server URL

    Notes:
        - Cost: $0 (run locally)
        - Privacy: 100% local, no data sent externally
        - Models: llama3.1, mistral, codellama, etc.
        - Requires: Ollama server running
        - Install: https://ollama.ai
    """

    POPULAR_MODELS = {
        "llama3.1:70b": "Llama 3.1 70B (best quality)",
        "llama3.1:8b": "Llama 3.1 8B (fast, good)",
        "mistral:7b": "Mistral 7B (balanced)",
        "codellama:13b": "CodeLlama 13B (code-focused)",
    }

    def __init__(self, host: str = "http://localhost:11434", model: str = "mistral:7b"):
        """Initialize Ollama analyzer.

        Args:
            host: Ollama server URL
            model: Ollama model name
        """
        super().__init__()
        self.provider = AIProvider.OLLAMA
        self.model = model
        self.host = host
        self.api_url = f"{host}/api/generate"
        logger.info(f"OllamaAnalyzer initialized with model {model} at {host}")

    def analyze_session(
        self, prompt: str, dataset: dict | None = None, *, system_prompt: str | None = None
    ) -> str:
        """Analyze session using Ollama local LLM.

        Args:
            prompt: Structured analysis prompt
            dataset: Optional dataset (unused)
            system_prompt: Optional system-level prompt. Ollama does not
                support message roles natively, so it is prepended to the
                user prompt when provided.

        Returns:
            AI-generated analysis markdown

        Raises:
            WorkflowError: If Ollama server is inaccessible
        """
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        logger.info(f"Sending prompt to Ollama ({len(full_prompt)} chars, model: {self.model})")

        try:
            response = requests.post(
                self.api_url,
                json={"model": self.model, "prompt": full_prompt, "stream": False},
                timeout=600,
            )
            response.raise_for_status()
            result = response.json()
            analysis = result.get("response", "")
            logger.info(f"Received analysis from Ollama ({len(analysis)} chars)")
            return analysis

        except requests.exceptions.ConnectionError:
            error_msg = (
                f"Cannot connect to Ollama server at {self.host}. "
                "Make sure Ollama is running: https://ollama.ai"
            )
            logger.error(error_msg)
            raise WorkflowError(error_msg) from None

        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise WorkflowError(f"Failed to analyze session with Ollama: {e}") from e

    def get_provider_info(self) -> dict:
        """Get Ollama provider info."""
        return {
            "provider": "ollama",
            "model": self.model,
            "host": self.host,
            "status": "ready" if self.validate_config() else "server_offline",
            "cost_input": "$0.00 (local)",
            "cost_output": "$0.00 (local)",
            "requires_api_key": False,
            "privacy": "100% local",
            "note": "Free, unlimited, private",
        }

    def validate_config(self) -> bool:
        """Validate Ollama server is running."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
