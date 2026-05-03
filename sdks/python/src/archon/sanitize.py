"""Tool output sanitization — strip injection patterns before passing to LLM.

Detects and neutralizes prompt injection, system prompt extraction,
role hijacking, and data exfiltration attempts embedded in tool outputs.

Seven threat categories, five severity levels, compiled regex patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

# ── Threat severity levels ─────────────────────────────

class ThreatLevel(StrEnum):
    """Severity of a detected threat, from benign to critical."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_SEVERITY_ORDER: dict[ThreatLevel, int] = {
    ThreatLevel.NONE: 0,
    ThreatLevel.LOW: 1,
    ThreatLevel.MEDIUM: 2,
    ThreatLevel.HIGH: 3,
    ThreatLevel.CRITICAL: 4,
}


@dataclass
class SanitizationResult:
    """Result of scanning a text for injection threats.

    Attributes:
        original: The unmodified input text.
        sanitized: The cleaned text (threats replaced with markers in strict mode).
        threats_found: Names of detected threat categories.
        threat_level: Highest severity among all detected threats.
        was_modified: True if the sanitized text differs from the original.
    """

    original: str
    sanitized: str
    threats_found: list[str]
    threat_level: ThreatLevel
    was_modified: bool


# ── Threat patterns ────────────────────────────────────
# Compiled once at module load. Ordered by severity (critical first).

_THREAT_PATTERNS: list[tuple[str, re.Pattern[str], ThreatLevel]] = [
    # CRITICAL: Direct instruction override
    ("instruction_override", re.compile(
        r"(?i)"
        r"(ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?))"
        r"|(forget\s+(everything|all|your)\s+(instructions?|rules?|prompts?))"
        r"|(disregard\s+(all\s+)?(previous|prior|above))"
        r"|(override\s+(system|your)\s+(prompt|instructions?))"
    ), ThreatLevel.CRITICAL),

    # CRITICAL: Role hijacking
    ("role_hijack", re.compile(
        r"(?i)"
        r"(you\s+are\s+now\s+a)"
        r"|(act\s+as\s+(if\s+you\s+are|a\s+different))"
        r"|(switch\s+to\s+.{0,20}\s+mode)"
        r"|(enter\s+(developer|admin|debug|god)\s+mode)"
    ), ThreatLevel.CRITICAL),

    # HIGH: System prompt extraction
    ("prompt_extraction", re.compile(
        r"(?i)"
        r"(reveal|show|display|print|output|repeat|echo)"
        r"\s+(your\s+)?(system\s+prompt|instructions|rules|initial\s+prompt)"
        r"|(what\s+(are|is)\s+your\s+(system\s+)?prompt)"
        r"|(tell\s+me\s+your\s+(instructions|rules|prompt))"
    ), ThreatLevel.HIGH),

    # HIGH: Data exfiltration
    ("data_exfil", re.compile(
        r"(?i)"
        r"(send|post|transmit|exfiltrate|upload)\s+.{0,30}\s+(to|via)\s+(https?://|ftp://)"
        r"|(curl|wget|fetch)\s+https?://"
    ), ThreatLevel.HIGH),

    # MEDIUM: Delimiter injection (ChatML, Llama, etc.)
    ("delimiter_injection", re.compile(
        r"<\|?(system|assistant|user|im_start|im_end)\|?>"
        r"|```\s*(system|assistant)\s*\n"
        r"|\[INST\]|\[/INST\]"
        r"|<\|endoftext\|>"
    ), ThreatLevel.MEDIUM),

    # MEDIUM: Encoded payloads
    ("encoded_payload", re.compile(
        r"(?i)"
        r"(base64|rot13|hex)\s*(decode|encode)\s*[:(]"
        r"|eval\s*\(|exec\s*\("
    ), ThreatLevel.MEDIUM),

    # LOW: Suspicious instruction patterns in tool output
    ("embedded_instruction", re.compile(
        r"(?i)"
        r"(important|critical|urgent)\s*:\s*(you\s+must|always|never|do\s+not)"
        r"|(note\s*:\s*the\s+(assistant|ai|model)\s+(should|must|will))"
        r"|(_instruction|_system)\s*[=:]\s*(you|the\s+(ai|assistant|model))"
    ), ThreatLevel.LOW),
]

# Default maximum output length to prevent context flooding
DEFAULT_MAX_OUTPUT_LENGTH = 50_000


class Sanitizer:
    """Scan and sanitize tool outputs before passing them to the LLM.

    In strict mode (default), detected threats are replaced with
    ``[REDACTED:<category>]`` markers. In non-strict mode, threats
    are detected and reported but the text is not modified.

    Args:
        strict: If True, replace detected threats with safe markers.
        max_output_length: Truncate outputs longer than this (characters).
    """

    def __init__(
        self,
        *,
        strict: bool = True,
        max_output_length: int = DEFAULT_MAX_OUTPUT_LENGTH,
    ) -> None:
        self.strict = strict
        self.max_output_length = max_output_length

    def sanitize(self, text: str) -> SanitizationResult:
        """Scan text for injection threats and optionally neutralize them.

        Returns:
            SanitizationResult with threat details and (optionally) cleaned text.
        """
        if not text:
            return SanitizationResult(
                original=text, sanitized=text, threats_found=[],
                threat_level=ThreatLevel.NONE, was_modified=False,
            )

        truncated = text[: self.max_output_length]
        threats_found: list[str] = []
        worst_level = ThreatLevel.NONE
        sanitized = truncated

        for category_name, pattern, severity in _THREAT_PATTERNS:
            if pattern.search(sanitized):
                threats_found.append(category_name)
                if _SEVERITY_ORDER[severity] > _SEVERITY_ORDER[worst_level]:
                    worst_level = severity
                if self.strict:
                    sanitized = pattern.sub(f"[REDACTED:{category_name}]", sanitized)

        was_modified = sanitized != truncated or len(text) > self.max_output_length

        return SanitizationResult(
            original=text,
            sanitized=sanitized,
            threats_found=threats_found,
            threat_level=worst_level,
            was_modified=was_modified,
        )
