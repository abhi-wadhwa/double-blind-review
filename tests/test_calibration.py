"""Tests for the calibration engine."""

import pytest

from src.core.calibration import CalibrationEngine
from src.core.models import Review


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_review(
    app_id: str,
    reviewer_id: str,
    scores: dict[str, float],
) -> Review:
    """Create a calibration review."""
    return Review(
        application_id=app_id,
        reviewer_id=reviewer_id,
        scores=scores,
        is_calibration=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCalibrationEngine:
    """Test suite for CalibrationEngine."""

    def test_perfect_agreement(self) -> None:
        """When all reviewers agree perfectly, deviations are zero."""
        reviews = [
            make_review("cal-1", "r1", {"quality": 4.0, "novelty": 3.0}),
            make_review("cal-1", "r2", {"quality": 4.0, "novelty": 3.0}),
            make_review("cal-1", "r3", {"quality": 4.0, "novelty": 3.0}),
        ]
        engine = CalibrationEngine(threshold=1.5)
        results = engine.analyze(reviews)

        for r in results:
            assert r.mean_deviation == 0.0
            assert not r.is_flagged

    def test_one_outlier_flagged(self) -> None:
        """A reviewer with very different scores should be flagged."""
        reviews = [
            # Three reviewers agree
            make_review("cal-1", "r1", {"quality": 4.0, "novelty": 4.0}),
            make_review("cal-1", "r2", {"quality": 4.0, "novelty": 4.0}),
            make_review("cal-1", "r3", {"quality": 4.0, "novelty": 4.0}),
            # One outlier
            make_review("cal-1", "r4", {"quality": 1.0, "novelty": 1.0}),
            # Second calibration app - same pattern
            make_review("cal-2", "r1", {"quality": 3.0, "novelty": 3.0}),
            make_review("cal-2", "r2", {"quality": 3.0, "novelty": 3.0}),
            make_review("cal-2", "r3", {"quality": 3.0, "novelty": 3.0}),
            make_review("cal-2", "r4", {"quality": 1.0, "novelty": 1.0}),
        ]
        engine = CalibrationEngine(threshold=1.0)
        results = engine.analyze(reviews)

        # r4 should be flagged
        r4_result = next(r for r in results if r.reviewer_id == "r4")
        assert r4_result.is_flagged
        assert r4_result.mean_deviation > 0

        # Others should not be flagged
        for r in results:
            if r.reviewer_id != "r4":
                assert not r.is_flagged

    def test_dimension_deviations_present(self) -> None:
        """Results should include per-dimension deviation data."""
        reviews = [
            make_review("cal-1", "r1", {"quality": 5.0, "novelty": 2.0}),
            make_review("cal-1", "r2", {"quality": 3.0, "novelty": 4.0}),
        ]
        engine = CalibrationEngine()
        results = engine.analyze(reviews)

        for r in results:
            assert "quality" in r.dimension_deviations
            assert "novelty" in r.dimension_deviations

    def test_empty_reviews(self) -> None:
        """Empty input should return empty results."""
        engine = CalibrationEngine()
        assert engine.analyze([]) == []

    def test_single_reviewer(self) -> None:
        """A single reviewer has zero deviation from self."""
        reviews = [
            make_review("cal-1", "r1", {"quality": 3.0}),
        ]
        engine = CalibrationEngine()
        results = engine.analyze(reviews)
        assert len(results) == 1
        # Single reviewer cannot be compared, deviation is 0
        assert results[0].mean_deviation == 0.0

    def test_consensus_scores(self) -> None:
        """Consensus should be the mean of reviewer scores."""
        reviews = [
            make_review("cal-1", "r1", {"quality": 2.0, "novelty": 4.0}),
            make_review("cal-1", "r2", {"quality": 4.0, "novelty": 2.0}),
        ]
        engine = CalibrationEngine()
        consensus = engine.get_consensus_scores(reviews)

        assert consensus["cal-1"]["quality"] == pytest.approx(3.0)
        assert consensus["cal-1"]["novelty"] == pytest.approx(3.0)

    def test_multiple_calibration_apps(self) -> None:
        """Engine handles multiple calibration applications."""
        reviews = [
            make_review("cal-1", "r1", {"q": 3.0}),
            make_review("cal-1", "r2", {"q": 4.0}),
            make_review("cal-2", "r1", {"q": 2.0}),
            make_review("cal-2", "r2", {"q": 3.0}),
            make_review("cal-3", "r1", {"q": 5.0}),
            make_review("cal-3", "r2", {"q": 4.0}),
        ]
        engine = CalibrationEngine()
        results = engine.analyze(reviews)
        assert len(results) == 2

    def test_invalid_threshold(self) -> None:
        """Negative threshold should raise ValueError."""
        with pytest.raises(ValueError):
            CalibrationEngine(threshold=-1.0)

    def test_sorted_by_deviation(self) -> None:
        """Results are sorted by mean deviation, highest first."""
        reviews = [
            make_review("cal-1", "r1", {"q": 5.0}),
            make_review("cal-1", "r2", {"q": 3.0}),
            make_review("cal-1", "r3", {"q": 1.0}),
        ]
        engine = CalibrationEngine()
        results = engine.analyze(reviews)

        devs = [r.mean_deviation for r in results]
        assert devs == sorted(devs, reverse=True)
