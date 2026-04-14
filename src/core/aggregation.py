"""Score aggregation with outlier handling.

Supports three aggregation methods:

- **Weighted average**: Each reviewer's score is weighted by the
  reviewer's calibration weight.
- **Trimmed mean**: The highest and lowest scores for each dimension
  are dropped before averaging (requires >= 3 reviews).
- **Median**: The median score per dimension.

Outlier handling: scores that deviate from the per-dimension mean by
more than ``outlier_sigma`` standard deviations are excluded before
aggregation.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from src.core.audit import AuditTrail
from src.core.models import AggregatedScore, AggregationMethod, Review, Reviewer
from src.core.rubric import RubricSystem


class ScoreAggregator:
    """Aggregate multiple reviewer scores into final application scores.

    Parameters
    ----------
    rubric:
        The rubric used for scoring.
    method:
        Aggregation strategy.
    outlier_sigma:
        Scores deviating more than this many standard deviations from
        the dimension mean are excluded.  Set to 0 to disable outlier
        removal.
    """

    def __init__(
        self,
        rubric: RubricSystem,
        method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
        outlier_sigma: float = 2.0,
        audit: AuditTrail | None = None,
    ) -> None:
        self.rubric = rubric
        self.method = method
        self.outlier_sigma = outlier_sigma
        self._audit = audit or AuditTrail()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _std(values: list[float], mean: float) -> float:
        if len(values) < 2:
            return 0.0
        return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

    def _remove_outliers(
        self,
        scores: list[float],
    ) -> tuple[list[float], int]:
        """Remove values more than ``outlier_sigma`` from the mean.

        Returns the cleaned list and the count of removed outliers.
        """
        if self.outlier_sigma <= 0 or len(scores) < 3:
            return scores, 0

        mean = self._mean(scores)
        std = self._std(scores, mean)

        if std == 0:
            return scores, 0

        cleaned = [
            s for s in scores if abs(s - mean) <= self.outlier_sigma * std
        ]
        return cleaned, len(scores) - len(cleaned)

    def _aggregate_values(
        self,
        values: list[float],
        weights: list[float],
    ) -> float:
        """Aggregate a list of values using the configured method."""
        if not values:
            return 0.0

        if self.method == AggregationMethod.MEDIAN:
            return float(statistics.median(values))

        if self.method == AggregationMethod.TRIMMED_MEAN:
            if len(values) >= 3:
                sorted_vals = sorted(values)
                trimmed = sorted_vals[1:-1]
                return self._mean(trimmed)
            return self._mean(values)

        # Weighted average (default)
        if not weights or all(w == 0 for w in weights):
            return self._mean(values)
        total_weight = sum(weights)
        if total_weight == 0:
            return self._mean(values)
        return sum(v * w for v, w in zip(values, weights)) / total_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate_application(
        self,
        reviews: list[Review],
        reviewers: list[Reviewer] | None = None,
        anonymous_id: str = "",
    ) -> AggregatedScore:
        """Aggregate all reviews for a single application.

        Parameters
        ----------
        reviews:
            All submitted reviews for one application.
        reviewers:
            Reviewer objects (used for weights in weighted-average mode).
        anonymous_id:
            The anonymous identifier for the application.

        Returns
        -------
        AggregatedScore
            The final aggregated result.
        """
        if not reviews:
            return AggregatedScore(
                anonymous_id=anonymous_id,
                application_id=reviews[0].application_id if reviews else "",
            )

        app_id = reviews[0].application_id
        reviewer_weight_map: dict[str, float] = {}
        if reviewers:
            reviewer_weight_map = {r.reviewer_id: r.weight for r in reviewers}

        # Collect scores per dimension
        dim_scores: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for review in reviews:
            w = reviewer_weight_map.get(review.reviewer_id, 1.0)
            for dim_name, score in review.scores.items():
                dim_scores[dim_name].append((score, w))

        dimension_results: dict[str, float] = {}
        total_outliers = 0

        for dim_name, score_weight_pairs in dim_scores.items():
            raw_scores = [s for s, _ in score_weight_pairs]
            weights = [w for _, w in score_weight_pairs]

            cleaned, removed = self._remove_outliers(raw_scores)
            total_outliers += removed

            # Rebuild weights for cleaned scores
            if removed > 0:
                cleaned_weights = []
                mean = self._mean(raw_scores)
                std = self._std(raw_scores, mean)
                for s, w in score_weight_pairs:
                    if std == 0 or abs(s - mean) <= self.outlier_sigma * std:
                        cleaned_weights.append(w)
                weights = cleaned_weights

            dimension_results[dim_name] = self._aggregate_values(
                cleaned, weights
            )

        # Compute overall score using rubric weights
        overall = self.rubric.compute_weighted_score(dimension_results)

        # Confidence: based on number of reviews and agreement
        num_reviews = len(reviews)
        if num_reviews >= 2:
            dim_stds = []
            for dim_name, score_weight_pairs in dim_scores.items():
                vals = [s for s, _ in score_weight_pairs]
                dim = (
                    self.rubric.get_dimension(dim_name)
                    if dim_name in self.rubric.dimension_names
                    else None
                )
                if dim and len(vals) >= 2:
                    span = dim.max_score - dim.min_score
                    if span > 0:
                        normalized_std = self._std(vals, self._mean(vals)) / span
                        dim_stds.append(normalized_std)
            avg_std = self._mean(dim_stds) if dim_stds else 0.0
            confidence = max(0.0, min(1.0, 1.0 - avg_std * 2))
        else:
            confidence = 0.5

        result = AggregatedScore(
            application_id=app_id,
            anonymous_id=anonymous_id,
            dimension_scores=dimension_results,
            overall_score=overall,
            num_reviews=num_reviews,
            num_outliers_removed=total_outliers,
            method=self.method,
            confidence=round(confidence, 3),
        )

        self._audit.log(
            action="aggregate",
            entity_id=app_id,
            details={
                "overall_score": round(overall, 4),
                "num_reviews": num_reviews,
                "outliers_removed": total_outliers,
                "method": self.method.value,
            },
        )

        return result

    def aggregate_all(
        self,
        reviews: list[Review],
        reviewers: list[Reviewer] | None = None,
        anon_id_map: dict[str, str] | None = None,
    ) -> list[AggregatedScore]:
        """Aggregate reviews for all applications and produce a ranked list.

        Parameters
        ----------
        reviews:
            All reviews across all applications.
        reviewers:
            Reviewer objects for weight lookup.
        anon_id_map:
            Mapping of ``application_id`` to ``anonymous_id``.

        Returns
        -------
        list[AggregatedScore]
            Results sorted by overall score (highest first), with
            ``rank`` fields populated.
        """
        anon_id_map = anon_id_map or {}

        by_app: dict[str, list[Review]] = defaultdict(list)
        for review in reviews:
            by_app[review.application_id].append(review)

        results: list[AggregatedScore] = []
        for app_id, app_reviews in by_app.items():
            anon_id = anon_id_map.get(app_id, "")
            result = self.aggregate_application(app_reviews, reviewers, anon_id)
            results.append(result)

        # Rank by overall score descending
        results.sort(key=lambda r: r.overall_score, reverse=True)
        for i, result in enumerate(results, start=1):
            result.rank = i

        return results
