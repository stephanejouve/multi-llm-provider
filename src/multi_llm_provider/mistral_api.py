"""Mistral AI provider implementation.

Cost-effective Mistral AI provider for analysis.
Best price/performance ratio.
"""

import logging

from mistralai.client.sdk import Mistral

from .base import AIAnalyzer, AIProvider, WorkflowError

logger = logging.getLogger(__name__)


class MistralAPIAnalyzer(AIAnalyzer):
    """Mistral AI API provider for cost-effective analysis.

    Attributes:
        provider: AIProvider.MISTRAL
        model: Mistral model name
        client: Mistral API client

    Notes:
        - Cost: ~$2/1M input tokens, $6/1M output tokens
        - Quality: comparable to GPT-4
        - Context: 32k tokens
    """

    MODELS = {
        "mistral-large-latest": "Large (best performance, $2/1M)",
        "mistral-medium-latest": "Medium (balanced, $1.5/1M)",
        "mistral-small-latest": "Small (fast, $0.5/1M)",
        "open-mistral-7b": "7B (free tier, basic)",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-large-latest",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        timeout: int = 60,
    ):
        """Initialize Mistral API analyzer.

        Args:
            api_key: Mistral API key
            model: Mistral model name
            temperature: Sampling temperature (0.0-1.0, default: 0.7)
            max_tokens: Max tokens in response (default: 4000)
            timeout: Request timeout in seconds (default: 60)

        Raises:
            WorkflowError: If API key is invalid
        """
        super().__init__()
        self.provider = AIProvider.MISTRAL
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        if not api_key:
            raise WorkflowError("Mistral API key required")

        try:
            self.client = Mistral(api_key=api_key)
            logger.info(
                f"MistralAPIAnalyzer initialized: model={model}, "
                f"temperature={temperature}, max_tokens={max_tokens}"
            )
        except Exception as e:
            raise WorkflowError(f"Failed to initialize Mistral API client: {e}") from e

    def analyze_session(
        self, prompt: str, dataset: dict | None = None, *, system_prompt: str | None = None
    ) -> str:
        """Analyze session using Mistral API.

        Args:
            prompt: Structured analysis prompt
            dataset: Optional dataset (unused)
            system_prompt: Optional system-level prompt. When provided,
                added as a system message. No default system message is used.

        Returns:
            AI-generated analysis markdown

        Raises:
            WorkflowError: If API call fails
        """
        logger.info(
            f"Sending prompt to Mistral API ({len(prompt)} chars, "
            f"model={self.model}, temperature={self.temperature})"
        )

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            analysis = response.choices[0].message.content

            # Handle Mistral API returning list[str] or str
            if isinstance(analysis, list):
                analysis = " ".join(analysis)

            logger.info(
                f"Received analysis from Mistral ({len(analysis)} chars, "
                f"tokens: ~{len(analysis) // 4})"
            )
            return analysis

        except Exception as e:
            logger.error(f"Mistral API call failed: {e}")
            raise WorkflowError(f"Failed to analyze session with Mistral API: {e}") from e

    def get_provider_info(self) -> dict:
        """Get Mistral API provider info."""
        return {
            "provider": "mistral_api",
            "model": self.model,
            "status": "ready",
            "cost_input": "$2.00/1M tokens",
            "cost_output": "$6.00/1M tokens",
            "requires_api_key": True,
            "context_window": "32k tokens",
            "note": "Best price/performance ratio",
        }

    def validate_config(self) -> bool:
        """Validate Mistral API configuration."""
        return self.client is not None
