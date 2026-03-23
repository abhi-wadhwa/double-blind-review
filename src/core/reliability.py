"""Inter-rater reliability using Krippendorff's alpha.

Krippendorff's alpha is a statistical measure of agreement among
multiple raters who assign categorical, ordinal, interval, or ratio
ratings to items.  It generalizes several specialized agreement
coefficients (Scott's pi, Cohen's kappa, etc.) and handles:

- Any number of raters
- Missing data (not all raters need to rate every item)
- Multiple data types (nominal, ordinal, interval, ratio)

Formula
-------
For *N* items rated by *R* raters on a scale with possible values *v*:

    alpha = 1 - D_o / D_e

where:
    D_o = observed disagreement (within-unit variance)
    D_e = expected disagreement (across all units)

For ordinal data, the difference function uses cumulative frequencies.
For interval/ratio data, it uses squared differences.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

from src.core.audit import AuditTrail
from src.core.models import Review


class ReliabilityCalculator:
    """Compute Krippendorff's alpha and related inter-rater statistics.

    Parameters
    ----------
    data_type:
        The measurement level: ``"nominal"``, ``"ordinal"``,
        ``"interval"``, or ``"ratio"``.
    """

    VALID_TYPES = {"nominal", "ordinal", "interval", "ratio"}

    def __init__(
        self,
        data_type: str = "interval",
        audit: Optional[AuditTrail] = None,
    ) -> None:
        if data_type not in self.VALID_TYPES:
            raise ValueError(
                f"data_type must be one of {self.VALID_TYPES}, got '{data_type}'"
            )
        self.data_type = data_type
        self._audit = audit or AuditTrail()

    # ------------------------------------------------------------------
    # Difference functions
    # ------------------------------------------------------------------

    def _delta_nominal(self, v1: float, v2: float) -> float:
        """Nominal difference: 0 if equal, 1 otherwise."""
        return 0.0 if v1 == v2 else 1.0

    def _delta_ordinal(
        self, v1: float, v2: float, value_counts: dict[float, int], sorted_values: list[float]
    ) -> float:
        """Ordinal difference based on cumulative frequency."""
        if v1 == v2:
            return 0.0
        lo, hi = (min(v1, v2), max(v1, v2))
        lo_idx = sorted_values.index(lo)
        hi_idx = sorted_values.index(hi)

        # Sum of counts between lo and hi (inclusive)
        cum_sum = sum(
            value_counts.get(sorted_values[i], 0)
            for i in range(lo_idx, hi_idx + 1)
        )
        # Subtract half of boundary counts
        cum_sum -= (value_counts.get(lo, 0) + value_counts.get(hi, 0)) / 2.0
        return cum_sum ** 2

    def _delta_interval(self, v1: float, v2: float) -> float:
        """Interval difference: squared difference."""
        return (v1 - v2) ** 2

    def _delta_ratio(self, v1: float, v2: float) -> float:
        """Ratio difference: squared difference relative to sum."""
        denom = v1 + v2
        if denom == 0:
            return 0.0
        return ((v1 - v2) / denom) ** 2

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute_alpha(
        self,
        reliability_data: dict[str, dict[str, float]],
    ) -> float:
        """Compute Krippendorff's alpha from a reliability data matrix.

        Parameters
        ----------
        reliability_data:
            ``{item_id: {rater_id: value}}``.  Not all raters need to
            rate every item.

        Returns
        -------
        float
            Alpha coefficient.  1.0 = perfect agreement, 0.0 = agreement
            at chance level, negative = below chance.
        """
        # Collect all values and their frequencies
        all_values: list[float] = []
        for rater_values in reliability_data.values():
            all_values.extend(rater_values.values())

        if not all_values:
            return 0.0

        n_total = len(all_values)
        if n_total < 2:
            return 0.0

        value_counts: dict[float, int] = defaultdict(int)
        for v in all_values:
            value_counts[v] += 1
        sorted_values = sorted(value_counts.keys())

        # Choose delta function
        if self.data_type == "nominal":
            delta = lambda a, b: self._delta_nominal(a, b)
        elif self.data_type == "ordinal":
            delta = lambda a, b: self._delta_ordinal(a, b, value_counts, sorted_values)
        elif self.data_type == "ratio":
            delta = lambda a, b: self._delta_ratio(a, b)
        else:
            delta = lambda a, b: self._delta_interval(a, b)

        # Compute observed disagreement D_o
        d_o = 0.0
        total_pairs_observed = 0

        for item_id, rater_values in reliability_data.items():
            values = list(rater_values.values())
            m_u = len(values)  # number of ratings for this unit
            if m_u < 2:
                continue
            for i in range(m_u):
                for j in range(i + 1, m_u):
                    d_o += delta(values[i], values[j])
                    total_pairs_observed += 1

            # Weight by 1/(m_u - 1) for each unit
            # Actually, the standard formulation weights the within-unit
            # sum by 1/(m_u - 1).  We accumulate and normalize below.

        # Recompute D_o using the proper Krippendorff weighting
        d_o_weighted = 0.0
        n_pairable = 0  # total number of pairable values

        for item_id, rater_values in reliability_data.items():
            values = list(rater_values.values())
            m_u = len(values)
            if m_u < 2:
                continue
            n_pairable += m_u
            unit_disagreement = 0.0
            for i in range(m_u):
                for j in range(i + 1, m_u):
                    unit_disagreement += delta(values[i], values[j])
            # Each unit weighted by 1/(m_u - 1)
            d_o_weighted += unit_disagreement / (m_u - 1)

        if n_pairable == 0:
            return 0.0

        d_o_final = d_o_weighted / n_pairable

        # Compute expected disagreement D_e
        d_e = 0.0
        n_pairs = 0
        unique_vals = sorted_values
        for i, v1 in enumerate(unique_vals):
            for j, v2 in enumerate(unique_vals):
                if i < j:
                    n_c = value_counts[v1]
                    n_k = value_counts[v2]
                    d_e += n_c * n_k * delta(v1, v2)
                    n_pairs += n_c * n_k

        if n_pairs == 0:
            return 1.0

        d_e_final = d_e / (n_total * (n_total - 1))

        if d_e_final == 0:
            return 1.0

        alpha = 1.0 - d_o_final / d_e_final
        return alpha

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_from_reviews(
        self,
        reviews: list[Review],
        dimension: Optional[str] = None,
    ) -> dict[str, float]:
        """Compute Krippendorff's alpha from review objects.

        Parameters
        ----------
        reviews:
            All reviews to include in the computation.
        dimension:
            If specified, compute alpha for that single dimension.
            If None, compute alpha for each dimension independently.

        Returns
        -------
        dict
            ``{dimension_name: alpha}``
        """
        # Collect all dimensions
        all_dims: set[str] = set()
        for review in reviews:
            all_dims.update(review.scores.keys())

        if dimension:
            all_dims = {dimension} if dimension in all_dims else set()

        results: dict[str, float] = {}

        for dim in sorted(all_dims):
            # Build reliability data matrix
            reliability_data: dict[str, dict[str, float]] = defaultdict(dict)
            for review in reviews:
                if dim in review.scores:
                    reliability_data[review.application_id][
                        review.reviewer_id
                    ] = review.scores[dim]

            alpha = self._compute_alpha(dict(reliability_data))
            results[dim] = round(alpha, 4)

            self._audit.log(
                action="reliability",
                entity_id=dim,
                details={"alpha": round(alpha, 4), "data_type": self.data_type},
            )

        return results

    def compute_overall_alpha(self, reviews: list[Review]) -> float:
        """Compute a single overall alpha across all dimensions.

        This treats each (application, dimension) pair as a separate
        item, giving a holistic measure of inter-rater agreement.
        """
        reliability_data: dict[str, dict[str, float]] = defaultdict(dict)

        for review in reviews:
            for dim, score in review.scores.items():
                item_key = f"{review.application_id}::{dim}"
                reliability_data[item_key][review.reviewer_id] = score

        alpha = self._compute_alpha(dict(reliability_data))

        self._audit.log(
            action="reliability_overall",
            entity_id="all",
            details={"alpha": round(alpha, 4)},
        )

        return round(alpha, 4)

    @staticmethod
    def interpret_alpha(alpha: float) -> str:
        """Return a human-readable interpretation of alpha.

        Uses Krippendorff's recommended thresholds:
        - alpha >= 0.800 : reliable
        - 0.667 <= alpha < 0.800 : tentative conclusions
        - alpha < 0.667 : unreliable
        """
        if alpha >= 0.800:
            return "Reliable agreement"
        if alpha >= 0.667:
            return "Tentative agreement (draw only tentative conclusions)"
        if alpha >= 0.0:
            return "Unreliable agreement (discard or re-examine)"
        return "Below-chance agreement (systematic disagreement)"
