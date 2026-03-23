"""Calibration engine for reviewer consistency analysis.

During a calibration round, all reviewers score the same set of
*calibration applications*.  The engine then computes each reviewer's
deviation from the consensus (mean score across all reviewers) and
flags reviewers whose scores diverge beyond a configurable threshold.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

from src.core.audit import AuditTrail
from src.core.models import CalibrationResult, Review


class CalibrationEngine:
    """Analyse calibration-round scores to detect reviewer inconsistency.

    Algorithm
    ---------
    For each calibration application and each dimension:

    1. Compute the consensus score (mean of all reviewers).
    2. For each reviewer, compute the signed deviation from the consensus.
    3. Across all calibration applications, compute the reviewer's
       mean absolute deviation and standard deviation.
    4. Flag the reviewer if their mean absolute deviation exceeds
       ``threshold`` times the pooled standard deviation of all reviewers.

    Parameters
    ----------
    threshold:
        Number of standard deviations above the mean at which a reviewer
        is flagged.  Default is 1.5.
    """

    def __init__(
        self,
        threshold: float = 1.5,
        audit: Optional[AuditTrail] = None,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold
        self._audit = audit or AuditTrail()

    def analyze(
        self,
        calibration_reviews: list[Review],
    ) -> list[CalibrationResult]:
        """Run calibration analysis on a set of calibration-round reviews.

        Parameters
        ----------
        calibration_reviews:
            Reviews submitted during the calibration round.  All reviews
            should have ``is_calibration=True`` and score the same set
            of dimensions.

        Returns
        -------
        list[CalibrationResult]
            One result per unique reviewer, sorted by mean deviation
            (highest first).
        """
        if not calibration_reviews:
            return []

        # Group reviews by application
        by_app: dict[str, list[Review]] = defaultdict(list)
        for review in calibration_reviews:
            by_app[review.application_id].append(review)

        # Collect all dimensions and all reviewers
        all_dimensions: set[str] = set()
        all_reviewer_ids: set[str] = set()
        for review in calibration_reviews:
            all_dimensions.update(review.scores.keys())
            all_reviewer_ids.add(review.reviewer_id)

        # Compute consensus (mean) for each (application, dimension)
        consensus: dict[str, dict[str, float]] = {}
        for app_id, reviews in by_app.items():
            consensus[app_id] = {}
            for dim in all_dimensions:
                scores = [r.scores[dim] for r in reviews if dim in r.scores]
                if scores:
                    consensus[app_id][dim] = sum(scores) / len(scores)

        # Compute per-reviewer deviations
        reviewer_deviations: dict[str, list[float]] = defaultdict(list)
        reviewer_dim_devs: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for app_id, reviews in by_app.items():
            for review in reviews:
                for dim in all_dimensions:
                    if dim in review.scores and dim in consensus.get(app_id, {}):
                        dev = abs(review.scores[dim] - consensus[app_id][dim])
                        reviewer_deviations[review.reviewer_id].append(dev)
                        reviewer_dim_devs[review.reviewer_id][dim].append(dev)

        # Compute pooled standard deviation across all reviewer mean deviations
        all_mean_devs: list[float] = []
        reviewer_mean_dev: dict[str, float] = {}
        reviewer_std_dev: dict[str, float] = {}

        for rid in all_reviewer_ids:
            devs = reviewer_deviations.get(rid, [])
            if devs:
                mean_dev = sum(devs) / len(devs)
                std_dev = math.sqrt(
                    sum((d - mean_dev) ** 2 for d in devs) / len(devs)
                ) if len(devs) > 1 else 0.0
            else:
                mean_dev = 0.0
                std_dev = 0.0
            reviewer_mean_dev[rid] = mean_dev
            reviewer_std_dev[rid] = std_dev
            all_mean_devs.append(mean_dev)

        # Pooled statistics for flagging
        if all_mean_devs:
            pooled_mean = sum(all_mean_devs) / len(all_mean_devs)
            pooled_std = math.sqrt(
                sum((d - pooled_mean) ** 2 for d in all_mean_devs)
                / len(all_mean_devs)
            ) if len(all_mean_devs) > 1 else 0.0
        else:
            pooled_mean = 0.0
            pooled_std = 0.0

        flag_cutoff = pooled_mean + self.threshold * pooled_std

        # Build results
        results: list[CalibrationResult] = []
        for rid in all_reviewer_ids:
            dim_mean_devs = {}
            for dim, devs in reviewer_dim_devs.get(rid, {}).items():
                dim_mean_devs[dim] = sum(devs) / len(devs) if devs else 0.0

            mean_dev = reviewer_mean_dev[rid]
            std_dev = reviewer_std_dev[rid]

            is_flagged = mean_dev > flag_cutoff if pooled_std > 0 else False
            flag_reason = ""
            if is_flagged:
                flag_reason = (
                    f"Mean deviation {mean_dev:.3f} exceeds threshold "
                    f"{flag_cutoff:.3f} (pooled mean {pooled_mean:.3f} + "
                    f"{self.threshold} * pooled std {pooled_std:.3f})"
                )

            results.append(
                CalibrationResult(
                    reviewer_id=rid,
                    mean_deviation=mean_dev,
                    std_deviation=std_dev,
                    dimension_deviations=dim_mean_devs,
                    is_flagged=is_flagged,
                    flag_reason=flag_reason,
                )
            )

            self._audit.log(
                action="calibrate",
                entity_id=rid,
                details={
                    "mean_deviation": round(mean_dev, 4),
                    "flagged": is_flagged,
                },
            )

        results.sort(key=lambda r: r.mean_deviation, reverse=True)
        return results

    def get_consensus_scores(
        self,
        calibration_reviews: list[Review],
    ) -> dict[str, dict[str, float]]:
        """Return consensus (mean) scores for each calibration application.

        Returns
        -------
        dict
            ``{application_id: {dimension: mean_score}}``
        """
        by_app: dict[str, list[Review]] = defaultdict(list)
        for review in calibration_reviews:
            by_app[review.application_id].append(review)

        consensus: dict[str, dict[str, float]] = {}
        for app_id, reviews in by_app.items():
            consensus[app_id] = {}
            all_dims: set[str] = set()
            for r in reviews:
                all_dims.update(r.scores.keys())
            for dim in all_dims:
                scores = [r.scores[dim] for r in reviews if dim in r.scores]
                if scores:
                    consensus[app_id][dim] = sum(scores) / len(scores)

        return consensus
