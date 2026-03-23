"""Tests for the score aggregation engine."""

import pytest

from src.core.aggregation import ScoreAggregator
from src.core.models import AggregatedScore, AggregationMethod, Review, Reviewer
from src.core.rubric import Dimension, RubricSystem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rubric() -> RubricSystem:
    """Simple 2-dimension rubric for testing."""
    return RubricSystem(
        dimensions=[
            Dimension(name="quality", min_score=1, max_score=5, weight=2.0),
            Dimension(name="clarity", min_score=1, max_score=5, weight=1.0),
        ]
    )


def make_review(
    app_id: str,
    reviewer_id: str,
    quality: float,
    clarity: float,
) -> Review:
    return Review(
        application_id=app_id,
        reviewer_id=reviewer_id,
        scores={"quality": quality, "clarity": clarity},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScoreAggregator:
    """Test suite for ScoreAggregator."""

    def test_weighted_average_basic(self, rubric: RubricSystem) -> None:
        """Weighted average with equal-weight reviewers."""
        reviews = [
            make_review("a1", "r1", 4.0, 3.0),
            make_review("a1", "r2", 5.0, 4.0),
        ]
        agg = ScoreAggregator(rubric, method=AggregationMethod.WEIGHTED_AVERAGE)
        result = agg.aggregate_application(reviews)

        # quality avg = 4.5, clarity avg = 3.5
        assert result.dimension_scores["quality"] == pytest.approx(4.5)
        assert result.dimension_scores["clarity"] == pytest.approx(3.5)
        assert result.num_reviews == 2

    def test_weighted_average_with_weights(self, rubric: RubricSystem) -> None:
        """Weighted average respects reviewer weights."""
        reviews = [
            make_review("a1", "r1", 4.0, 4.0),
            make_review("a1", "r2", 2.0, 2.0),
        ]
        reviewers = [
            Reviewer(reviewer_id="r1", weight=3.0),
            Reviewer(reviewer_id="r2", weight=1.0),
        ]
        agg = ScoreAggregator(rubric, method=AggregationMethod.WEIGHTED_AVERAGE)
        result = agg.aggregate_application(reviews, reviewers)

        # weighted quality = (4*3 + 2*1) / 4 = 3.5
        assert result.dimension_scores["quality"] == pytest.approx(3.5)

    def test_trimmed_mean(self, rubric: RubricSystem) -> None:
        """Trimmed mean drops highest and lowest."""
        reviews = [
            make_review("a1", "r1", 1.0, 1.0),  # low - trimmed
            make_review("a1", "r2", 3.0, 3.0),
            make_review("a1", "r3", 4.0, 4.0),
            make_review("a1", "r4", 5.0, 5.0),  # high - trimmed
        ]
        agg = ScoreAggregator(rubric, method=AggregationMethod.TRIMMED_MEAN)
        result = agg.aggregate_application(reviews)

        # After trimming: quality = (3+4)/2 = 3.5, clarity same
        assert result.dimension_scores["quality"] == pytest.approx(3.5)
        assert result.dimension_scores["clarity"] == pytest.approx(3.5)

    def test_trimmed_mean_with_two_reviews(self, rubric: RubricSystem) -> None:
        """Trimmed mean falls back to regular mean when < 3 reviews."""
        reviews = [
            make_review("a1", "r1", 2.0, 2.0),
            make_review("a1", "r2", 4.0, 4.0),
        ]
        agg = ScoreAggregator(rubric, method=AggregationMethod.TRIMMED_MEAN)
        result = agg.aggregate_application(reviews)

        # Falls back to mean: (2+4)/2 = 3.0
        assert result.dimension_scores["quality"] == pytest.approx(3.0)

    def test_median(self, rubric: RubricSystem) -> None:
        """Median aggregation selects the middle value."""
        reviews = [
            make_review("a1", "r1", 1.0, 1.0),
            make_review("a1", "r2", 3.0, 3.0),
            make_review("a1", "r3", 5.0, 5.0),
        ]
        agg = ScoreAggregator(rubric, method=AggregationMethod.MEDIAN)
        result = agg.aggregate_application(reviews)

        assert result.dimension_scores["quality"] == pytest.approx(3.0)
        assert result.dimension_scores["clarity"] == pytest.approx(3.0)

    def test_outlier_removal(self, rubric: RubricSystem) -> None:
        """Outlier beyond 2 sigma should be removed."""
        reviews = [
            make_review("a1", "r1", 4.0, 4.0),
            make_review("a1", "r2", 4.0, 4.0),
            make_review("a1", "r3", 4.0, 4.0),
            make_review("a1", "r4", 1.0, 1.0),  # outlier
        ]
        agg = ScoreAggregator(
            rubric,
            method=AggregationMethod.WEIGHTED_AVERAGE,
            outlier_sigma=1.0,  # strict threshold
        )
        result = agg.aggregate_application(reviews)

        # With outlier removed, quality should be closer to 4.0
        assert result.dimension_scores["quality"] > 3.5
        assert result.num_outliers_removed >= 1

    def test_outlier_removal_disabled(self, rubric: RubricSystem) -> None:
        """Setting sigma=0 disables outlier removal."""
        reviews = [
            make_review("a1", "r1", 4.0, 4.0),
            make_review("a1", "r2", 4.0, 4.0),
            make_review("a1", "r3", 1.0, 1.0),
        ]
        agg = ScoreAggregator(rubric, outlier_sigma=0)
        result = agg.aggregate_application(reviews)
        assert result.num_outliers_removed == 0

    def test_handles_missing_reviews(self, rubric: RubricSystem) -> None:
        """Aggregation works when a reviewer scored only some dimensions."""
        reviews = [
            Review(
                application_id="a1",
                reviewer_id="r1",
                scores={"quality": 4.0},  # missing clarity
            ),
            Review(
                application_id="a1",
                reviewer_id="r2",
                scores={"quality": 3.0, "clarity": 5.0},
            ),
        ]
        agg = ScoreAggregator(rubric)
        result = agg.aggregate_application(reviews)

        assert result.dimension_scores["quality"] == pytest.approx(3.5)
        # clarity only from r2
        assert result.dimension_scores["clarity"] == pytest.approx(5.0)

    def test_aggregate_all_ranking(self, rubric: RubricSystem) -> None:
        """aggregate_all should return results ranked by overall score."""
        reviews = [
            make_review("a1", "r1", 2.0, 2.0),
            make_review("a1", "r2", 2.0, 2.0),
            make_review("a2", "r1", 5.0, 5.0),
            make_review("a2", "r2", 5.0, 5.0),
            make_review("a3", "r1", 3.0, 3.0),
            make_review("a3", "r2", 3.0, 3.0),
        ]
        agg = ScoreAggregator(rubric)
        results = agg.aggregate_all(reviews)

        assert results[0].application_id == "a2"  # highest
        assert results[0].rank == 1
        assert results[-1].application_id == "a1"  # lowest
        assert results[-1].rank == 3

    def test_aggregate_all_with_anon_ids(self, rubric: RubricSystem) -> None:
        """Anonymous IDs should propagate through aggregation."""
        reviews = [
            make_review("a1", "r1", 3.0, 3.0),
        ]
        agg = ScoreAggregator(rubric)
        results = agg.aggregate_all(
            reviews, anon_id_map={"a1": "Applicant-001"}
        )
        assert results[0].anonymous_id == "Applicant-001"

    def test_confidence_high_agreement(self, rubric: RubricSystem) -> None:
        """High agreement should yield high confidence."""
        reviews = [
            make_review("a1", "r1", 4.0, 4.0),
            make_review("a1", "r2", 4.0, 4.0),
            make_review("a1", "r3", 4.0, 4.0),
        ]
        agg = ScoreAggregator(rubric)
        result = agg.aggregate_application(reviews)
        assert result.confidence >= 0.9

    def test_confidence_low_agreement(self, rubric: RubricSystem) -> None:
        """Low agreement should yield lower confidence."""
        reviews = [
            make_review("a1", "r1", 1.0, 1.0),
            make_review("a1", "r2", 5.0, 5.0),
        ]
        agg = ScoreAggregator(rubric)
        result = agg.aggregate_application(reviews)
        assert result.confidence < 0.9

    def test_ties_in_ranking(self, rubric: RubricSystem) -> None:
        """Tied scores should both receive unique ranks (no gaps)."""
        reviews = [
            make_review("a1", "r1", 3.0, 3.0),
            make_review("a2", "r1", 3.0, 3.0),
        ]
        agg = ScoreAggregator(rubric)
        results = agg.aggregate_all(reviews)

        ranks = [r.rank for r in results]
        assert sorted(ranks) == [1, 2]
