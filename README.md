# multi-llm-provider

Unified multi-provider LLM interface supporting Claude, OpenAI, Mistral, Ollama, and Clipboard.

## Installation

```bash
pip install multi-llm-provider
```

### Optional providers

Install only the providers you need:

```bash
pip install multi-llm-provider[claude]    # Anthropic Claude
pip install multi-llm-provider[openai]    # OpenAI GPT
pip install multi-llm-provider[mistral]   # Mistral AI
pip install multi-llm-provider[ollama]    # Ollama (local)
pip install multi-llm-provider[all]       # All providers
```

## Quick start

```python
from multi_llm_provider import AIProvider, AIProviderFactory

# Create an analyzer for Claude
analyzer = AIProviderFactory.create(
    provider=AIProvider.CLAUDE_API,
    model="claude-sonnet-4-20250514",
    system_prompt="You are a helpful assistant.",
)

# Analyze content
result = analyzer.analyze("Explain quantum computing in simple terms.")
print(result)
```

## Supported providers

| Provider | Enum value | Required extra |
|----------|-----------|----------------|
| Anthropic Claude | `AIProvider.CLAUDE_API` | `claude` |
| OpenAI GPT | `AIProvider.OPENAI_API` | `openai` |
| Mistral AI | `AIProvider.MISTRAL_API` | `mistral` |
| Ollama (local) | `AIProvider.OLLAMA` | `ollama` |
| Clipboard (manual) | `AIProvider.CLIPBOARD` | *none* |

## Architecture

All providers implement the `AIAnalyzer` abstract base class:

```python
from multi_llm_provider import AIAnalyzer

class AIAnalyzer(ABC):
    @abstractmethod
    def analyze(self, content: str) -> str:
        """Send content to the LLM and return the response."""
        ...
```

The `AIProviderFactory` creates the appropriate analyzer based on the provider enum:

```python
analyzer = AIProviderFactory.create(
    provider=AIProvider.OPENAI_API,
    model="gpt-4o",
    system_prompt="You are a data analyst.",
    temperature=0.7,
    max_tokens=4096,
)
```

### Clipboard provider

The `ClipboardAnalyzer` copies content to the system clipboard with instructions,
then waits for the user to paste the LLM response back. Useful for manual workflows
or when API access is not available.

## Error handling

```python
from multi_llm_provider import ConfigError, WorkflowError

try:
    result = analyzer.analyze("some content")
except WorkflowError as e:
    print(f"LLM call failed: {e}")
except ConfigError as e:
    print(f"Configuration error: {e}")
```

## Development

```bash
git clone https://github.com/stephanejouve/multi-llm-provider.git
cd multi-llm-provider
poetry install --with dev,extras
poetry run pytest tests/ -v
poetry run black src/ tests/ --check --line-length=100
poetry run ruff check src/
poetry run isort src/ tests/ --check-only --profile black --line-length=100
```

## License

MIT
