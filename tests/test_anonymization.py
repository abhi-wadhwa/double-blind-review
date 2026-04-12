"""Tests for the anonymization engine."""

import re

import pytest

from src.core.anonymization import AnonymizationEngine, RedactionStats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PII_CORPUS = [
    {
        "text": (
            "Dear Committee, my name is Dr. John Smith and I work at MIT. "
            "Please contact me at john.smith@mit.edu or call 617-555-1234."
        ),
        "expect_redacted": ["email", "phone", "name"],
    },
    {
        "text": (
            "Prof. Maria Garcia from Stanford University presents her work on "
            "NLP. Visit https://mariagarcia.stanford.edu for more details."
        ),
        "expect_redacted": ["name", "institution", "url"],
    },
    {
        "text": (
            "Applicant: Ms. Jane Doe, Harvard College. "
            "SSN: 123-45-6789. Phone: (555) 987-6543. "
            "Email: jane.doe@harvard.edu"
        ),
        "expect_redacted": ["email", "phone", "ssn", "name", "institution"],
    },
    {
        "text": "No PII here, just a plain application about machine learning.",
        "expect_redacted": [],
    },
    {
        "text": (
            "Dr. Robert Chen\nUniversity of California Berkeley\n"
            "robert.chen@berkeley.edu\n+1-510-555-9999\n"
            "http://robchen.com/cv"
        ),
        "expect_redacted": ["email", "phone", "url", "name", "institution"],
    },
]


@pytest.fixture
def engine() -> AnonymizationEngine:
    return AnonymizationEngine()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnonymizationEngine:
    """Test suite for AnonymizationEngine."""

    def test_email_redaction(self, engine: AnonymizationEngine) -> None:
        text = "Contact me at alice@example.com or bob.jones@university.edu."
        result, _, stats = engine.anonymize(text, "t1")
        assert "alice@example.com" not in result
        assert "bob.jones@university.edu" not in result
        assert "[EMAIL_REDACTED]" in result
        assert stats.emails_removed == 2

    def test_phone_redaction(self, engine: AnonymizationEngine) -> None:
        text = "Call 555-123-4567 or (800) 555-0199 or +1-212-555-0000."
        result, _, stats = engine.anonymize(text, "t2")
        assert "555-123-4567" not in result
        assert "(800) 555-0199" not in result
        assert stats.phones_removed >= 2

    def test_url_redaction(self, engine: AnonymizationEngine) -> None:
        text = "See https://example.com/my-cv and http://portfolio.io/work."
        result, _, stats = engine.anonymize(text, "t3")
        assert "https://example.com" not in result
        assert "http://portfolio.io" not in result
        assert stats.urls_removed == 2

    def test_name_redaction(self, engine: AnonymizationEngine) -> None:
        text = "Written by Dr. Alice Wonderland and Prof. Bob Builder."
        result, _, stats = engine.anonymize(text, "t4")
        assert "Alice Wonderland" not in result
        assert "Bob Builder" not in result
        assert stats.names_removed == 2

    def test_institution_redaction(self, engine: AnonymizationEngine) -> None:
        text = "Graduated from Stanford University and Harvard College."
        result, _, stats = engine.anonymize(text, "t5")
        assert "Stanford University" not in result
        assert "Harvard College" not in result
        assert stats.institutions_removed >= 1

    def test_ssn_redaction(self, engine: AnonymizationEngine) -> None:
        text = "SSN: 123-45-6789 and 987 65 4321."
        result, _, stats = engine.anonymize(text, "t6")
        assert "123-45-6789" not in result
        assert stats.ssns_removed >= 1

    def test_anonymous_id_assignment(self, engine: AnonymizationEngine) -> None:
        _, id1, _ = engine.anonymize("Text one", "app-1")
        _, id2, _ = engine.anonymize("Text two", "app-2")
        _, id1_again, _ = engine.anonymize("Text one v2", "app-1")

        assert id1 == "Applicant-001"
        assert id2 == "Applicant-002"
        assert id1_again == id1  # stable mapping

    def test_no_pii_in_corpus(self, engine: AnonymizationEngine) -> None:
        """Ensure no PII leaks across the entire test corpus."""
        for case in PII_CORPUS:
            result, _, _ = engine.anonymize(case["text"], f"corpus-{id(case)}")
            remaining = engine.verify_no_pii(result)
            assert remaining == [], (
                f"PII still present: {remaining}\n"
                f"Original: {case['text']}\n"
                f"Anonymized: {result}"
            )

    def test_verify_clean_text(self, engine: AnonymizationEngine) -> None:
        clean = "This text has no personal information at all."
        assert engine.verify_no_pii(clean) == []

    def test_redaction_stats_total(self) -> None:
        stats = RedactionStats(
            emails_removed=2, phones_removed=1, names_removed=3
        )
        assert stats.total == 6

    def test_batch_anonymize(self, engine: AnonymizationEngine) -> None:
        docs = [
            ("Email: a@b.com", "d1"),
            ("Call 555-111-2222", "d2"),
        ]
        results = engine.anonymize_batch(docs)
        assert len(results) == 2
        assert results[0][1] == "Applicant-001"
        assert results[1][1] == "Applicant-002"

    def test_custom_pattern(self) -> None:
        custom = [re.compile(r"\bProject\s+\w+\b")]
        engine = AnonymizationEngine(custom_patterns=custom)
        result, _, stats = engine.anonymize("Working on Project Alpha now.", "c1")
        assert "Project Alpha" not in result
        assert stats.custom_removed == 1

    def test_reset(self, engine: AnonymizationEngine) -> None:
        engine.anonymize("text", "a1")
        assert len(engine.id_map) == 1
        engine.reset()
        assert len(engine.id_map) == 0

    def test_empty_text(self, engine: AnonymizationEngine) -> None:
        result, anon_id, stats = engine.anonymize("", "empty")
        assert result == ""
        assert anon_id == "Applicant-001"
        assert stats.total == 0
