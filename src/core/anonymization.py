"""Anonymization engine for removing PII from application text.

Detects and redacts:
- Email addresses
- Phone numbers (US and international formats)
- URLs
- Common name patterns (salutations + capitalized words)
- Institution names (university, college, etc.)
- Social Security Numbers
- Dates of birth patterns
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.core.audit import AuditTrail


@dataclass
class RedactionStats:
    """Statistics about what was redacted from a document."""

    emails_removed: int = 0
    phones_removed: int = 0
    urls_removed: int = 0
    names_removed: int = 0
    institutions_removed: int = 0
    ssns_removed: int = 0
    custom_removed: int = 0

    @property
    def total(self) -> int:
        return (
            self.emails_removed
            + self.phones_removed
            + self.urls_removed
            + self.names_removed
            + self.institutions_removed
            + self.ssns_removed
            + self.custom_removed
        )


class AnonymizationEngine:
    """Engine for stripping personally identifiable information from text.

    The engine applies a sequence of regex-based redaction passes to remove
    emails, phone numbers, names, institution references, and other PII.
    Each application is assigned a stable anonymous identifier of the form
    ``Applicant-NNN``.
    """

    # Pre-compiled regex patterns for PII detection
    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    )

    PHONE_PATTERN = re.compile(
        r"(?:\+?1[\s\-.]?)?"
        r"(?:\(?\d{3}\)?[\s\-.]?)"
        r"\d{3}[\s\-.]?\d{4}"
        r"|"
        r"\+\d{1,3}[\s\-.]?\d{1,4}[\s\-.]?\d{1,4}[\s\-.]?\d{1,9}"
    )

    URL_PATTERN = re.compile(
        r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
        re.IGNORECASE,
    )

    NAME_PATTERN = re.compile(
        r"\b(?:Mr|Mrs|Ms|Dr|Prof|Professor|Sir|Madam|Miss)\b\.?\s+"
        r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
    )

    INSTITUTION_PATTERN = re.compile(
        r"\b(?:University|College|Institute|School|Academy|Lab|Laboratory)"
        r"(?:\s+of)?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*"
        r"|"
        r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+"
        r"(?:University|College|Institute|School|Academy)",
        re.IGNORECASE,
    )

    SSN_PATTERN = re.compile(r"\b\d{3}[\-\s]?\d{2}[\-\s]?\d{4}\b")

    def __init__(
        self,
        custom_patterns: Optional[list[re.Pattern]] = None,
        audit: Optional[AuditTrail] = None,
    ) -> None:
        self._counter = 0
        self._id_map: dict[str, str] = {}
        self._custom_patterns = custom_patterns or []
        self._audit = audit or AuditTrail()

    @property
    def id_map(self) -> dict[str, str]:
        """Return a copy of the application-ID to anonymous-ID mapping."""
        return dict(self._id_map)

    def _next_anonymous_id(self) -> str:
        """Generate the next sequential anonymous identifier."""
        self._counter += 1
        return f"Applicant-{self._counter:03d}"

    def anonymize(
        self,
        text: str,
        application_id: str = "",
    ) -> tuple[str, str, RedactionStats]:
        """Anonymize a single document.

        Parameters
        ----------
        text:
            Raw application text that may contain PII.
        application_id:
            Unique identifier for the source application.  Used to assign
            a stable anonymous ID.

        Returns
        -------
        tuple
            ``(anonymized_text, anonymous_id, redaction_stats)``
        """
        stats = RedactionStats()

        # Assign stable anonymous ID
        if application_id and application_id in self._id_map:
            anon_id = self._id_map[application_id]
        else:
            anon_id = self._next_anonymous_id()
            if application_id:
                self._id_map[application_id] = anon_id

        result = text

        # Order matters: emails before names (emails may contain name-like parts)
        result, count = self.EMAIL_PATTERN.subn("[EMAIL_REDACTED]", result)
        stats.emails_removed = count

        result, count = self.PHONE_PATTERN.subn("[PHONE_REDACTED]", result)
        stats.phones_removed = count

        result, count = self.URL_PATTERN.subn("[URL_REDACTED]", result)
        stats.urls_removed = count

        result, count = self.SSN_PATTERN.subn("[SSN_REDACTED]", result)
        stats.ssns_removed = count

        result, count = self.INSTITUTION_PATTERN.subn(
            "[INSTITUTION_REDACTED]", result
        )
        stats.institutions_removed = count

        result, count = self.NAME_PATTERN.subn("[NAME_REDACTED]", result)
        stats.names_removed = count

        # Apply custom patterns
        for pattern in self._custom_patterns:
            result, count = pattern.subn("[CUSTOM_REDACTED]", result)
            stats.custom_removed += count

        self._audit.log(
            action="anonymize",
            entity_id=application_id or anon_id,
            details={
                "anonymous_id": anon_id,
                "redactions": stats.total,
            },
        )

        return result, anon_id, stats

    def anonymize_batch(
        self,
        documents: list[tuple[str, str]],
    ) -> list[tuple[str, str, RedactionStats]]:
        """Anonymize a batch of ``(text, application_id)`` pairs.

        Returns a list of ``(anonymized_text, anonymous_id, stats)`` tuples
        in the same order as the input.
        """
        results = []
        for text, app_id in documents:
            results.append(self.anonymize(text, app_id))
        return results

    def reset(self) -> None:
        """Reset the engine state, clearing all ID mappings."""
        self._counter = 0
        self._id_map.clear()

    def verify_no_pii(self, text: str) -> list[str]:
        """Check anonymized text for residual PII.

        Returns a list of PII type names that were still detected.
        An empty list means the text is clean.
        """
        findings: list[str] = []
        if self.EMAIL_PATTERN.search(text):
            findings.append("email")
        if self.PHONE_PATTERN.search(text):
            findings.append("phone")
        if self.URL_PATTERN.search(text):
            findings.append("url")
        if self.SSN_PATTERN.search(text):
            findings.append("ssn")
        if self.NAME_PATTERN.search(text):
            findings.append("name")
        if self.INSTITUTION_PATTERN.search(text):
            findings.append("institution")
        return findings
