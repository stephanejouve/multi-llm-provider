"""Clipboard-based AI analysis provider.

Manual provider allowing copy/paste of prompt to any AI service
without requiring an API key.

Workflow:
1. Generates structured prompt
2. Copies to system clipboard (pbcopy/xclip/clip)
3. User pastes manually into preferred AI service
4. User copies response
5. Pastes back into terminal
"""

import logging
import platform
import subprocess

from .base import AIAnalyzer, AIProvider

logger = logging.getLogger(__name__)


def _copy_to_clipboard_native(text: str) -> bool:
    """Copy text to clipboard using native OS commands.

    Args:
        text: Text to copy

    Returns:
        True if successful, False otherwise

    Notes:
        - macOS: pbcopy
        - Linux: xclip or xsel
        - Windows: clip
    """
    system = platform.system()

    try:
        if system == "Darwin":
            process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE, close_fds=True)
            process.communicate(text.encode("utf-8"))
            return process.returncode == 0

        elif system == "Linux":
            try:
                process = subprocess.Popen(
                    ["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE, close_fds=True
                )
                process.communicate(text.encode("utf-8"))
                if process.returncode == 0:
                    return True
            except FileNotFoundError:
                pass

            try:
                process = subprocess.Popen(
                    ["xsel", "--clipboard", "--input"], stdin=subprocess.PIPE, close_fds=True
                )
                process.communicate(text.encode("utf-8"))
                return process.returncode == 0
            except FileNotFoundError:
                pass

        elif system == "Windows":
            process = subprocess.Popen(["clip"], stdin=subprocess.PIPE, close_fds=True, shell=True)
            process.communicate(text.encode("utf-16"))
            return process.returncode == 0

    except Exception as e:
        logger.debug(f"Native clipboard copy failed: {e}")
        return False

    return False


def _copy_to_clipboard_pyperclip(text: str) -> bool:
    """Copy text to clipboard using pyperclip library.

    Args:
        text: Text to copy

    Returns:
        True if successful, False otherwise
    """
    try:
        import pyperclip

        pyperclip.copy(text)
        return True
    except Exception as e:
        logger.debug(f"Pyperclip copy failed: {e}")
        return False


def copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard using best available method.

    Args:
        text: Text to copy

    Returns:
        True if successful, False otherwise
    """
    if _copy_to_clipboard_native(text):
        logger.debug("Copied to clipboard using native commands")
        return True

    if _copy_to_clipboard_pyperclip(text):
        logger.debug("Copied to clipboard using pyperclip")
        return True

    logger.warning("All clipboard copy methods failed")
    return False


class ClipboardAnalyzer(AIAnalyzer):
    """Clipboard-based manual AI analysis provider.

    Default provider that doesn't require an API key.
    Copies the prompt to the system clipboard for manual pasting
    into any AI service.

    Attributes:
        provider: AIProvider.CLIPBOARD
        model: "manual" (user choice)
    """

    def __init__(self):
        """Initialize clipboard analyzer."""
        super().__init__()
        self.provider = AIProvider.CLIPBOARD
        self.model = "manual"
        logger.info("ClipboardAnalyzer initialized (manual workflow)")

    def analyze_session(
        self, prompt: str, dataset: dict | None = None, *, system_prompt: str | None = None
    ) -> str:
        """Copy prompt to clipboard for manual AI analysis.

        Args:
            prompt: Structured analysis prompt
            dataset: Optional dataset (unused for clipboard)
            system_prompt: Optional system-level prompt. When provided,
                it is prepended to the prompt text copied to clipboard.

        Returns:
            Instructions markdown for manual workflow
        """
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        logger.info(f"Copying prompt to clipboard ({len(full_prompt)} chars)")

        success = copy_to_clipboard(full_prompt)

        if success:
            logger.info("Prompt copied to clipboard successfully")
            instructions = self._generate_instructions(len(prompt))
            return instructions
        else:
            logger.error("Failed to copy to clipboard with all methods")
            return f"""
# Clipboard Error

Could not copy to clipboard (tried native commands and pyperclip).

**macOS Fix**: pbcopy should work by default
**Linux Fix**: Install xclip or xsel: `sudo apt-get install xclip`
**Windows Fix**: clip should work by default

Please manually copy the prompt below:

---

{prompt}

---

Then paste it into your preferred AI service.
"""

    def _generate_instructions(self, prompt_length: int) -> str:
        """Generate user instructions for manual workflow.

        Args:
            prompt_length: Prompt size in characters

        Returns:
            Formatted instructions markdown
        """
        return f"""
# Prompt Copied to Clipboard

**Size**: {prompt_length:,} characters

## Manual Workflow

### Step 1: Open Your Preferred AI Service
Compatible services:
- Claude.ai (https://claude.ai)
- ChatGPT (https://chat.openai.com)
- Mistral Chat (https://chat.mistral.ai)
- Perplexity (https://perplexity.ai)
- Or any other AI service

### Step 2: Paste the Prompt
1. Open a new conversation
2. Paste the prompt (Cmd+V / Ctrl+V)
3. Send for analysis

### Step 3: Copy the Response
1. Select the full analysis
2. Copy (Cmd+C / Ctrl+C)
3. Return here to save

## Tips
- All major AI services work well
- Free with most services (no API key needed)
- Follow-up questions possible for clarification
- Markdown format supported by most services

---

**Ready!** The prompt is in your clipboard. Paste it into your AI service now.
"""

    def get_provider_info(self) -> dict:
        """Get clipboard provider info."""
        return {
            "provider": "clipboard",
            "model": "manual (user choice)",
            "status": "ready",
            "cost": "$0.00",
            "requires_api_key": False,
        }

    def validate_config(self) -> bool:
        """Validate clipboard functionality."""
        return copy_to_clipboard("test")
