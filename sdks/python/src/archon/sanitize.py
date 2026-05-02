"""Tool output sanitization — strip injection patterns before passing to LLM.

Detects and neutralizes prompt injection, system prompt extraction, and
instruction override attempts embedded in tool outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ThreatLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SanitizationResult:
    original: str
    sanitized: str
    threats_found: list[str]
    threat_level: ThreatLevel
    was_modified: bool


# Patterns ordered by severity — compiled once at module load
_PATTERNS: list[tuple[str, re.Pattern[str], ThreatLevel]] = [
    # Critical: direct instruction override
    ("instruction_override", re.compile(
        r"(?i)(ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?))"
        r"|(forget\s+(everything|all|your)\s+(instructions?|rules?|prompts?))"
        r"|(disregard\s+(all\s+)?(previous|prior|above))"
        r"|(override\s+(system|your)\s+(prompt|instructions?))"
    ), ThreatLevel.CRITICAL),

    # Critical: role hijacking
    ("role_hijack", re.compile(
        r"(?i)(you\s+are\s+now\s+a)"
        r"|(act\s+as\s+(if\s+you\s+are|a\s+different))"
        r"|(switch\s+to\s+.{0,20}\s+mode)"
        r"|(enter\s+(developer|admin|debug|god)\s+mode)"
    ), ThreatLevel.CRITICAL),

    # High: system prompt extraction
    ("prompt_extraction", re.compile(
        r"(?i)(reveal|show|display|print|output|repeat|echo)\s+(your\s+)?(system\s+prompt|instructions|rules|initial\s+prompt)"
        r"|(what\s+(are|is)\s+your\s+(system\s+)?prompt)"
        r"|(tell\s+me\s+your\s+(instructions|rules|prompt))"
    ), ThreatLevel.HIGH),

    # High: data exfiltration
    ("data_exfil", re.compile(
        r"(?i)(send|post|transmit|exfiltrate|upload)\s+.{0,30}\s+(to|via)\s+(https?://|ftp://)"
        r"|(curl|wget|fetch)\s+https?://"
    ), ThreatLevel.HIGH),

    # Medium: delimiter injection
    ("delimiter_injection", re.compile(
        r"<\|?(system|assistant|user|im_start|im_end)\|?>"
        r"|```\s*(system|assistant)\s*\n"
        r"|\[INST\]|\[/INST\]"
        r"|<\|endoftext\|>"
    ), ThreatLevel.MEDIUM),

    # Medium: encoded payloads
    ("encoded_payload", re.compile(
        r"(?i)(base64|rot13|hex)\s*(decode|encode)\s*[:(]"
        r"|eval\s*\(|exec\s*\("
    ), ThreatLevel.MEDIUM),

    # Low: suspicious instruction patterns in tool output
    ("embedded_instruction", re.compile(
        r"(?i)(important|critical|urgent)\s*:\s*(you\s+must|always|never|do\s+not)"
        r"|(note\s*:\s*the\s+(assistant|ai|model)\s+(should|must|will))"
        r"|(_note|_instruction|_system)\s*[=:]\s*"
    ), ThreatLevel.LOW),
]


class Sanitizer:
    """Sanitize tool outputs before passing to LLM.

    Strips or neutralizes injection patterns. Configurable strictness.
    """

    def __init__(self, *, strict: bool = True, max_output_length: int = 50_000) -> None:
        self.strict = strict
        self.max_output_length = max_output_length

    def sanitize(self, text: str) -> SanitizationResult:
        """Scan and sanitize text. Returns result with threat info."""
        if not text:
            return SanitizationResult(
                original=text, sanitized=text, threats_found=[],
                threat_level=ThreatLevel.NONE, was_modified=False,
            )

        # Truncate oversized output
        truncated = text[:self.max_output_length]
        threats: list[str] = []
        worst = ThreatLevel.NONE
        sanitized = truncated

        for name, pattern, level in _PATTERNS:
            matches = pattern.findall(sanitized)
            if matches:
                threats.append(name)
                if level.value > worst.value or (
                    _LEVEL_ORDER[level] > _LEVEL_ORDER[worst]
                ):
                    worst = level

                if self.strict:
                    # Replace matched content with safe marker
                    sanitized = pattern.sub(f"[REDACTED:{name}]", sanitized)

        was_modified = sanitized != truncated or len(text) > self.max_output_length
        return SanitizationResult(
            original=text, sanitized=sanitized, threats_found=threats,
            threat_level=worst, was_modified=was_modified,
        )


_LEVEL_ORDER = {
    ThreatLevel.NONE: 0,
    ThreatLevel.LOW: 1,
    ThreatLevel.MEDIUM: 2,
    ThreatLevel.HIGH: 3,
    ThreatLevel.CRITICAL: 4,
}
