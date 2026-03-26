"""Unified multi-provider LLM interface.

Supports Claude, OpenAI, Mistral, Ollama, and Clipboard providers
through a common abstract interface.
"""

from .base import AIAnalyzer, AIProvider, WorkflowError
from .factory import AIProviderFactory, ConfigError

__all__ = [
    "AIProvider",
    "AIAnalyzer",
    "AIProviderFactory",
    "ConfigError",
    "WorkflowError",
]

__version__ = "0.1.0"
