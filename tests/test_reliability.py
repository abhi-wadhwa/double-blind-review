"""Tests for Krippendorff's alpha inter-rater reliability."""

import pytest

from src.core.models import Review
from src.core.reliability import ReliabilityCalculator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_review(
    app_id: str,
    reviewer_id: str,
    scores: dict[str, float],
) -> Review:
    return Review(
        application_id=app_id,
        reviewer_id=reviewer_id,
        scores=scores,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReliabilityCalculator:
    """Test suite for Krippendorff's alpha calculation."""

    def test_perfect_agreement(self) -> None:
        """Identical scores across all raters => alpha = 1.0."""
        reviews = [
            make_review("a1", "r1", {"dim": 4.0}),
            make_review("a1", "r2", {"dim": 4.0}),
            make_review("a2", "r1", {"dim": 3.0}),
            make_review("a2", "r2", {"dim": 3.0}),
            make_review("a3", "r1", {"dim": 5.0}),
            make_review("a3", "r2", {"dim": 5.0}),
        ]
        calc = ReliabilityCalculator(data_type="interval")
        alphas = calc.compute_from_reviews(reviews)

        assert alphas["dim"] == pytest.approx(1.0, abs=0.01)

    def test_no_agreement(self) -> None:
        """Random-like disagreement => alpha near 0 or negative."""
        reviews = [
            make_review("a1", "r1", {"dim": 1.0}),
            make_review("a1", "r2", {"dim": 5.0}),
            make_review("a2", "r1", {"dim": 5.0}),
            make_review("a2", "r2", {"dim": 1.0}),
        ]
        calc = ReliabilityCalculator(data_type="interval")
        alphas = calc.compute_from_reviews(reviews)

        # Should be low (negative or near zero for systematic disagreement)
        assert alphas["dim"] < 0.5

    def test_nominal_data_type(self) -> None:
        """Alpha works with nominal data."""
        reviews = [
            make_review("a1", "r1", {"cat": 1.0}),
            make_review("a1", "r2", {"cat": 1.0}),
            make_review("a2", "r1", {"cat": 2.0}),
            make_review("a2", "r2", {"cat": 2.0}),
            make_review("a3", "r1", {"cat": 1.0}),
            make_review("a3", "r2", {"cat": 1.0}),
        ]
        calc = ReliabilityCalculator(data_type="nominal")
        alphas = calc.compute_from_reviews(reviews)

        assert alphas["cat"] == pytest.approx(1.0, abs=0.01)

    def test_ordinal_data_type(self) -> None:
        """Alpha works with ordinal data."""
        reviews = [
            make_review("a1", "r1", {"ord": 1.0}),
            make_review("a1", "r2", {"ord": 2.0}),
            make_review("a2", "r1", {"ord": 3.0}),
            make_review("a2", "r2", {"ord": 3.0}),
            make_review("a3", "r1", {"ord": 5.0}),
            make_review("a3", "r2", {"ord": 4.0}),
        ]
        calc = ReliabilityCalculator(data_type="ordinal")
        alphas = calc.compute_from_reviews(reviews)

        assert "ord" in alphas
        assert -1.0 <= alphas["ord"] <= 1.0

    def test_ratio_data_type(self) -> None:
        """Alpha works with ratio data."""
        reviews = [
            make_review("a1", "r1", {"val": 10.0}),
            make_review("a1", "r2", {"val": 10.0}),
            make_review("a2", "r1", {"val": 20.0}),
            make_review("a2", "r2", {"val": 20.0}),
        ]
        calc = ReliabilityCalculator(data_type="ratio")
        alphas = calc.compute_from_reviews(reviews)

        assert alphas["val"] == pytest.approx(1.0, abs=0.01)

    def test_multiple_dimensions(self) -> None:
        """Alpha is computed independently per dimension."""
        reviews = [
            make_review("a1", "r1", {"quality": 4.0, "clarity": 3.0}),
            make_review("a1", "r2", {"quality": 4.0, "clarity": 5.0}),
            make_review("a2", "r1", {"quality": 2.0, "clarity": 2.0}),
            make_review("a2", "r2", {"quality": 2.0, "clarity": 4.0}),
            make_review("a3", "r1", {"quality": 5.0, "clarity": 1.0}),
            make_review("a3", "r2", {"quality": 5.0, "clarity": 3.0}),
        ]
        calc = ReliabilityCalculator()
        alphas = calc.compute_from_reviews(reviews)

        assert "quality" in alphas
        assert "clarity" in alphas
        # Quality has perfect agreement, clarity does not
        assert alphas["quality"] > alphas["clarity"]

    def test_single_dimension_filter(self) -> None:
        """Can compute alpha for just one dimension."""
        reviews = [
            make_review("a1", "r1", {"q": 3.0, "c": 1.0}),
            make_review("a1", "r2", {"q": 3.0, "c": 5.0}),
            make_review("a2", "r1", {"q": 4.0, "c": 2.0}),
            make_review("a2", "r2", {"q": 4.0, "c": 3.0}),
        ]
        calc = ReliabilityCalculator()
        alphas = calc.compute_from_reviews(reviews, dimension="q")
        assert "q" in alphas
        assert "c" not in alphas

    def test_overall_alpha(self) -> None:
        """Overall alpha treats each (app, dim) as a separate item."""
        reviews = [
            make_review("a1", "r1", {"q": 4.0, "c": 4.0}),
            make_review("a1", "r2", {"q": 4.0, "c": 4.0}),
            make_review("a2", "r1", {"q": 3.0, "c": 3.0}),
            make_review("a2", "r2", {"q": 3.0, "c": 3.0}),
        ]
        calc = ReliabilityCalculator()
        overall = calc.compute_overall_alpha(reviews)
        assert overall == pytest.approx(1.0, abs=0.01)

    def test_missing_data(self) -> None:
        """Alpha handles cases where not all raters rated every item."""
        reviews = [
            make_review("a1", "r1", {"dim": 3.0}),
            make_review("a1", "r2", {"dim": 4.0}),
            make_review("a2", "r1", {"dim": 5.0}),
            # r2 did not rate a2
            make_review("a3", "r2", {"dim": 2.0}),
            make_review("a3", "r1", {"dim": 3.0}),
        ]
        calc = ReliabilityCalculator()
        alphas = calc.compute_from_reviews(reviews)

        assert "dim" in alphas
        assert -1.0 <= alphas["dim"] <= 1.0

    def test_empty_reviews(self) -> None:
        """Empty reviews return empty results."""
        calc = ReliabilityCalculator()
        assert calc.compute_from_reviews([]) == {}
        assert calc.compute_overall_alpha([]) == 0.0

    def test_invalid_data_type(self) -> None:
        """Invalid data type should raise ValueError."""
        with pytest.raises(ValueError):
            ReliabilityCalculator(data_type="invalid")

    def test_interpret_alpha(self) -> None:
        """Test interpretation thresholds."""
        calc = ReliabilityCalculator()
        assert "Reliable" in calc.interpret_alpha(0.85)
        assert "Tentative" in calc.interpret_alpha(0.75)
        assert "Unreliable" in calc.interpret_alpha(0.5)
        assert "Below-chance" in calc.interpret_alpha(-0.1)

    def test_three_raters(self) -> None:
        """Alpha with three raters."""
        reviews = [
            make_review("a1", "r1", {"dim": 3.0}),
            make_review("a1", "r2", {"dim": 3.0}),
            make_review("a1", "r3", {"dim": 4.0}),
            make_review("a2", "r1", {"dim": 5.0}),
            make_review("a2", "r2", {"dim": 5.0}),
            make_review("a2", "r3", {"dim": 5.0}),
            make_review("a3", "r1", {"dim": 1.0}),
            make_review("a3", "r2", {"dim": 2.0}),
            make_review("a3", "r3", {"dim": 1.0}),
        ]
        calc = ReliabilityCalculator()
        alphas = calc.compute_from_reviews(reviews)

        # Should show reasonable agreement
        assert alphas["dim"] > 0.5
