#!/usr/bin/env python3
"""Tests for Clipboard AI Provider."""

import subprocess
from unittest.mock import MagicMock, patch

from multi_llm_provider.base import AIProvider
from multi_llm_provider.clipboard import (
    ClipboardAnalyzer,
    _copy_to_clipboard_native,
    _copy_to_clipboard_pyperclip,
    copy_to_clipboard,
)


class TestClipboardAnalyzer:

    def test_initialization(self):
        analyzer = ClipboardAnalyzer()
        assert analyzer is not None
        assert analyzer.provider == AIProvider.CLIPBOARD

    @patch("multi_llm_provider.clipboard.copy_to_clipboard")
    def test_analyze_session_copies_to_clipboard(self, mock_copy):
        mock_copy.return_value = True
        analyzer = ClipboardAnalyzer()
        result = analyzer.analyze_session("Test prompt")
        mock_copy.assert_called_once_with("Test prompt")
        assert result is not None
        assert len(result) > 0

    @patch("multi_llm_provider.clipboard.copy_to_clipboard")
    def test_analyze_session_with_dataset(self, mock_copy):
        mock_copy.return_value = True
        analyzer = ClipboardAnalyzer()
        result = analyzer.analyze_session("Test", dataset={"key": "val"})
        mock_copy.assert_called_once_with("Test")
        assert result is not None

    def test_get_provider_info(self):
        analyzer = ClipboardAnalyzer()
        info = analyzer.get_provider_info()
        assert info["provider"] == "clipboard"
        assert "$0" in str(info).lower() or "free" in str(info).lower()

    @patch("multi_llm_provider.clipboard.copy_to_clipboard")
    def test_clipboard_copy_failure_handling(self, mock_copy):
        mock_copy.return_value = False
        analyzer = ClipboardAnalyzer()
        result = analyzer.analyze_session("Test")
        assert result is not None
        assert len(result) > 0

    @patch("multi_llm_provider.clipboard.copy_to_clipboard")
    def test_no_system_prompt_copies_prompt_only(self, mock_copy):
        mock_copy.return_value = True
        analyzer = ClipboardAnalyzer()
        analyzer.analyze_session("User prompt only")
        mock_copy.assert_called_once_with("User prompt only")

    @patch("multi_llm_provider.clipboard.copy_to_clipboard")
    def test_system_prompt_concatenated(self, mock_copy):
        mock_copy.return_value = True
        analyzer = ClipboardAnalyzer()
        analyzer.analyze_session("User data", system_prompt="Coach role")
        copied_text = mock_copy.call_args[0][0]
        assert "Coach role" in copied_text
        assert "User data" in copied_text


class TestCopyToClipboardNativeDarwin:

    @patch("multi_llm_provider.clipboard.platform.system", return_value="Darwin")
    @patch("multi_llm_provider.clipboard.subprocess.Popen")
    def test_darwin_pbcopy_success(self, mock_popen, _mock_sys):
        proc = MagicMock()
        proc.returncode = 0
        mock_popen.return_value = proc
        assert _copy_to_clipboard_native("hello") is True
        mock_popen.assert_called_once_with(["pbcopy"], stdin=subprocess.PIPE, close_fds=True)

    @patch("multi_llm_provider.clipboard.platform.system", return_value="Darwin")
    @patch("multi_llm_provider.clipboard.subprocess.Popen")
    def test_darwin_pbcopy_failure(self, mock_popen, _mock_sys):
        proc = MagicMock()
        proc.returncode = 1
        mock_popen.return_value = proc
        assert _copy_to_clipboard_native("hello") is False


class TestCopyToClipboardNativeLinux:

    @patch("multi_llm_provider.clipboard.platform.system", return_value="Linux")
    @patch("multi_llm_provider.clipboard.subprocess.Popen")
    def test_linux_xclip_success(self, mock_popen, _mock_sys):
        proc = MagicMock()
        proc.returncode = 0
        mock_popen.return_value = proc
        assert _copy_to_clipboard_native("data") is True
        assert mock_popen.call_count == 1

    @patch("multi_llm_provider.clipboard.platform.system", return_value="Linux")
    @patch("multi_llm_provider.clipboard.subprocess.Popen")
    def test_linux_both_missing(self, mock_popen, _mock_sys):
        mock_popen.side_effect = FileNotFoundError("not found")
        assert _copy_to_clipboard_native("data") is False


class TestCopyToClipboardNativeEdgeCases:

    @patch("multi_llm_provider.clipboard.platform.system", return_value="FreeBSD")
    def test_unknown_os_returns_false(self, _mock_sys):
        assert _copy_to_clipboard_native("data") is False


class TestCopyToClipboardPyperclip:

    def test_pyperclip_success(self):
        mock_pyperclip = MagicMock()
        with patch.dict("sys.modules", {"pyperclip": mock_pyperclip}):
            assert _copy_to_clipboard_pyperclip("hello") is True

    def test_pyperclip_import_fails(self):
        with patch.dict("sys.modules", {"pyperclip": None}):
            assert _copy_to_clipboard_pyperclip("hello") is False


class TestCopyToClipboardOrchestration:

    @patch("multi_llm_provider.clipboard._copy_to_clipboard_native", return_value=True)
    @patch("multi_llm_provider.clipboard._copy_to_clipboard_pyperclip")
    def test_native_success_skips_pyperclip(self, mock_pyp, mock_native):
        assert copy_to_clipboard("data") is True
        mock_native.assert_called_once_with("data")
        mock_pyp.assert_not_called()

    @patch("multi_llm_provider.clipboard._copy_to_clipboard_native", return_value=False)
    @patch("multi_llm_provider.clipboard._copy_to_clipboard_pyperclip", return_value=True)
    def test_native_fails_pyperclip_fallback(self, mock_pyp, mock_native):
        assert copy_to_clipboard("data") is True

    @patch("multi_llm_provider.clipboard._copy_to_clipboard_native", return_value=False)
    @patch("multi_llm_provider.clipboard._copy_to_clipboard_pyperclip", return_value=False)
    def test_all_methods_fail(self, mock_pyp, mock_native):
        assert copy_to_clipboard("data") is False
