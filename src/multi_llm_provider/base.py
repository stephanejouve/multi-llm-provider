"""Base classes and enums for AI analysis providers.

Defines abstract interface for multi-provider AI integration
(Clipboard, Claude API, OpenAI, Mistral AI, Ollama).
"""

from abc import ABC, abstractmethod
from enum import Enum


class WorkflowError(Exception):
    """Error raised during AI provider operations."""

    pass


class AIProvider(Enum):
    """Supported AI analysis providers.

    Attributes:
        CLIPBOARD: Manual copy/paste workflow (no API, free)
        CLAUDE: Claude API (Anthropic)
        OPENAI: OpenAI GPT-4 Turbo
        MISTRAL: Mistral AI API (best price/performance)
        OLLAMA: Local LLM server (free, privacy)

    Examples:
        >>> provider = AIProvider.CLIPBOARD
        >>> provider.value
        'clipboard'
    """

    CLIPBOARD = "clipboard"
    CLAUDE = "claude_api"
    OPENAI = "openai"
    MISTRAL = "mistral_api"
    OLLAMA = "ollama"


class AIAnalyzer(ABC):
    """Abstract base class for AI analysis providers.

    All AI implementations (Clipboard, Claude API, etc.)
    must inherit from this class and implement analyze_session().

    Attributes:
        provider: Provider type (AIProvider enum)
        model: Model name (optional)

    Examples:
        >>> analyzer = ClipboardAnalyzer()
        >>> result = analyzer.analyze_session(prompt)
        'Analysis copied to clipboard'
    """

    def __init__(self):
        """Initialize AI analyzer base."""
        self.provider: AIProvider | None = None
        self.model: str | None = None

    @abstractmethod
    def analyze_session(
        self, prompt: str, dataset: dict | None = None, *, system_prompt: str | None = None
    ) -> str:
        """Analyze session with AI provider.

        Args:
            prompt: Structured prompt markdown for AI analysis
            dataset: Optional session dataset for context
            system_prompt: Optional system-level prompt (role, constraints).
                Providers that support message roles (Claude, Mistral, OpenAI)
                will use this as a dedicated system message. Others will
                prepend it to the user prompt.

        Returns:
            AI-generated analysis as markdown string

        Raises:
            NotImplementedError: If method not implemented

        Examples:
            >>> result = analyzer.analyze_session(prompt)
            >>> print(result[:100])
            '# Session Analysis...'

        Notes:
            - Clipboard provider: copies prompt, returns instructions
            - API providers: send prompt, return response
            - Ollama: sends to local server, returns response
        """
        raise NotImplementedError("Subclass must implement analyze_session()")

    def get_provider_info(self) -> dict:
        """Get provider information.

        Returns:
            Dict with provider name, model, and status

        Examples:
            >>> info = analyzer.get_provider_info()
            >>> print(info['provider'])
            'clipboard'
        """
        return {
            "provider": self.provider.value if self.provider else "unknown",
            "model": self.model or "default",
            "status": "ready",
        }

    def validate_config(self) -> bool:
        """Validate provider configuration.

        Returns:
            True if configuration is valid, False otherwise

        Examples:
            >>> is_valid = analyzer.validate_config()
            >>> print(is_valid)
            True

        Notes:
            - Clipboard: always valid (no config)
            - API providers: check api_key is present
        """
        return True  # Override in subclasses if needed
