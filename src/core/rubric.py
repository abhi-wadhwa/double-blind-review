"""Rubric system for structured scoring dimensions.

Admins define a rubric as a collection of *dimensions*, each with a name,
description, numeric scale, and optional anchor descriptions for each
scale point.  Rubrics are validated and can be serialized / deserialized
for storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Dimension:
    """A single scoring dimension within a rubric.

    Parameters
    ----------
    name:
        Short machine-friendly name (e.g. ``"technical_merit"``).
    label:
        Human-readable label shown to reviewers.
    description:
        Longer description guiding scoring.
    min_score:
        Minimum score on the scale (inclusive).
    max_score:
        Maximum score on the scale (inclusive).
    weight:
        Relative weight of this dimension in the overall score.
    anchors:
        Optional mapping of score values to descriptive anchors
        (e.g. ``{1: "Poor", 3: "Average", 5: "Excellent"}``).
    """

    name: str
    label: str = ""
    description: str = ""
    min_score: float = 1.0
    max_score: float = 5.0
    weight: float = 1.0
    anchors: dict[float, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name.replace("_", " ").title()
        if self.min_score >= self.max_score:
            raise ValueError(
                f"min_score ({self.min_score}) must be less than "
                f"max_score ({self.max_score}) for dimension '{self.name}'"
            )
        if self.weight < 0:
            raise ValueError(
                f"weight must be non-negative for dimension '{self.name}'"
            )

    def validate_score(self, score: float) -> bool:
        """Return True if *score* is within the valid range."""
        return self.min_score <= score <= self.max_score

    def normalize(self, score: float) -> float:
        """Normalize *score* to [0, 1] range."""
        span = self.max_score - self.min_score
        if span == 0:
            return 0.0
        return (score - self.min_score) / span

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "weight": self.weight,
            "anchors": {str(k): v for k, v in self.anchors.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> Dimension:
        """Deserialize from a plain dictionary."""
        anchors = {float(k): v for k, v in data.get("anchors", {}).items()}
        return cls(
            name=data["name"],
            label=data.get("label", ""),
            description=data.get("description", ""),
            min_score=float(data.get("min_score", 1.0)),
            max_score=float(data.get("max_score", 5.0)),
            weight=float(data.get("weight", 1.0)),
            anchors=anchors,
        )


class RubricSystem:
    """Manages a set of scoring dimensions for a review cycle.

    A rubric consists of one or more :class:`Dimension` objects.  The system
    ensures that dimension names are unique and provides helpers for score
    validation and normalization.
    """

    def __init__(self, dimensions: Optional[list[Dimension]] = None) -> None:
        self._dimensions: dict[str, Dimension] = {}
        for dim in dimensions or []:
            self.add_dimension(dim)

    @property
    def dimensions(self) -> list[Dimension]:
        """Return all dimensions in insertion order."""
        return list(self._dimensions.values())

    @property
    def dimension_names(self) -> list[str]:
        """Return names of all dimensions."""
        return list(self._dimensions.keys())

    @property
    def total_weight(self) -> float:
        """Return sum of all dimension weights."""
        return sum(d.weight for d in self._dimensions.values())

    def add_dimension(self, dim: Dimension) -> None:
        """Add a new dimension to the rubric.

        Raises
        ------
        ValueError
            If a dimension with the same name already exists.
        """
        if dim.name in self._dimensions:
            raise ValueError(f"Dimension '{dim.name}' already exists in rubric")
        self._dimensions[dim.name] = dim

    def remove_dimension(self, name: str) -> None:
        """Remove a dimension by name."""
        if name not in self._dimensions:
            raise KeyError(f"Dimension '{name}' not found in rubric")
        del self._dimensions[name]

    def get_dimension(self, name: str) -> Dimension:
        """Retrieve a dimension by name."""
        if name not in self._dimensions:
            raise KeyError(f"Dimension '{name}' not found in rubric")
        return self._dimensions[name]

    def validate_scores(self, scores: dict[str, float]) -> list[str]:
        """Validate a complete set of scores against the rubric.

        Returns a list of error messages.  An empty list means all scores
        are valid.
        """
        errors: list[str] = []
        for name, dim in self._dimensions.items():
            if name not in scores:
                errors.append(f"Missing score for dimension '{name}'")
            elif not dim.validate_score(scores[name]):
                errors.append(
                    f"Score {scores[name]} for '{name}' is outside "
                    f"[{dim.min_score}, {dim.max_score}]"
                )
        for name in scores:
            if name not in self._dimensions:
                errors.append(f"Unknown dimension '{name}' in scores")
        return errors

    def compute_weighted_score(self, scores: dict[str, float]) -> float:
        """Compute the weighted overall score from dimension scores.

        Scores are normalized to [0, 1] and combined using dimension
        weights.  The result is also in [0, 1].
        """
        if not self._dimensions:
            return 0.0

        total_weight = self.total_weight
        if total_weight == 0:
            return 0.0

        weighted_sum = 0.0
        for name, dim in self._dimensions.items():
            if name in scores:
                weighted_sum += dim.normalize(scores[name]) * dim.weight

        return weighted_sum / total_weight

    def to_dict(self) -> dict:
        """Serialize the full rubric to a dictionary."""
        return {
            "dimensions": [d.to_dict() for d in self._dimensions.values()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> RubricSystem:
        """Deserialize a rubric from a dictionary."""
        dims = [Dimension.from_dict(d) for d in data.get("dimensions", [])]
        return cls(dimensions=dims)

    @classmethod
    def default_rubric(cls) -> RubricSystem:
        """Create a standard rubric suitable for academic applications."""
        return cls(
            dimensions=[
                Dimension(
                    name="technical_merit",
                    label="Technical Merit",
                    description="Quality and rigor of technical approach",
                    min_score=1,
                    max_score=5,
                    weight=2.0,
                    anchors={1: "Poor", 2: "Below Average", 3: "Average", 4: "Good", 5: "Excellent"},
                ),
                Dimension(
                    name="originality",
                    label="Originality",
                    description="Novelty and creativity of the work",
                    min_score=1,
                    max_score=5,
                    weight=1.5,
                    anchors={1: "Derivative", 3: "Incremental", 5: "Highly Original"},
                ),
                Dimension(
                    name="clarity",
                    label="Clarity of Presentation",
                    description="Writing quality and organization",
                    min_score=1,
                    max_score=5,
                    weight=1.0,
                    anchors={1: "Unclear", 3: "Adequate", 5: "Exceptionally Clear"},
                ),
                Dimension(
                    name="impact",
                    label="Potential Impact",
                    description="Significance and broader implications",
                    min_score=1,
                    max_score=5,
                    weight=1.5,
                    anchors={1: "Minimal", 3: "Moderate", 5: "Transformative"},
                ),
                Dimension(
                    name="feasibility",
                    label="Feasibility",
                    description="Likelihood of successful execution",
                    min_score=1,
                    max_score=5,
                    weight=1.0,
                    anchors={1: "Unrealistic", 3: "Plausible", 5: "Highly Feasible"},
                ),
            ]
        )
