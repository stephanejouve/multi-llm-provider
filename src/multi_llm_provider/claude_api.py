"""Claude API integration.

Anthropic Claude provider for AI analysis.

Examples:
    >>> from multi_llm_provider import AIProviderFactory
    >>> analyzer = AIProviderFactory.create('claude_api', {
    ...     'claude_api_key': 'sk-ant-...'
    ... })
"""

import logging

from anthropic import Anthropic

from .base import AIAnalyzer, AIProvider, WorkflowError

logger = logging.getLogger(__name__)


class ClaudeAPIAnalyzer(AIAnalyzer):
    """Claude API provider via Anthropic SDK.

    Attributes:
        provider: AIProvider.CLAUDE
        model: Claude model name
        client: Anthropic API client
        max_tokens: Maximum tokens in response

    Notes:
        - Cost: ~$3/1M input tokens, $15/1M output tokens
        - Quality: excellent
        - Context: 200k tokens
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4000,
    ):
        """Initialize Claude API analyzer.

        Args:
            api_key: Anthropic API key (must start with 'sk-ant-')
            model: Claude model name
            max_tokens: Maximum tokens in response

        Raises:
            WorkflowError: If API key format is invalid
        """
        super().__init__()
        self.provider = AIProvider.CLAUDE
        self.model: str = model
        self.max_tokens: int = max_tokens

        if not api_key or not api_key.startswith("sk-ant-"):
            raise WorkflowError("Invalid Claude API key format (must start with 'sk-ant-')")

        try:
            self.client = Anthropic(api_key=api_key)
            logger.info(f"ClaudeAPIAnalyzer initialized with model {model}")
        except Exception as e:
            raise WorkflowError(f"Failed to initialize Claude API client: {e}") from e

    def analyze_session(
        self, prompt: str, dataset: dict | None = None, *, system_prompt: str | None = None
    ) -> str:
        """Analyze session using Claude API.

        Args:
            prompt: Structured analysis prompt
            dataset: Optional dataset (unused, prompt already complete)
            system_prompt: Optional system-level prompt passed as Anthropic
                ``system`` parameter for dedicated role/context framing.

        Returns:
            AI-generated analysis markdown

        Raises:
            WorkflowError: If API call fails
        """
        logger.info(f"Sending prompt to Claude API ({len(prompt)} chars)")

        try:
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            response = self.client.messages.create(**kwargs)

            content = response.content[0]
            if not hasattr(content, "text"):
                raise WorkflowError("Unexpected response format: content has no text attribute")

            analysis = content.text
            logger.info(f"Received analysis from Claude ({len(analysis)} chars)")
            return analysis

        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise WorkflowError(f"Failed to analyze session with Claude API: {e}") from e

    def get_provider_info(self) -> dict:
        """Get Claude API provider info."""
        return {
            "provider": "claude_api",
            "model": self.model,
            "status": "ready",
            "cost_input": "$3.00/1M tokens",
            "cost_output": "$15.00/1M tokens",
            "requires_api_key": True,
            "context_window": "200k tokens",
        }

    def validate_config(self) -> bool:
        """Validate Claude API configuration."""
        return self.client is not None
