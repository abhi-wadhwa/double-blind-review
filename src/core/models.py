"""Data models for the double-blind review platform."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class ApplicationStatus(str, Enum):
    """Status of an application in the review pipeline."""

    SUBMITTED = "submitted"
    ANONYMIZED = "anonymized"
    UNDER_REVIEW = "under_review"
    REVIEWED = "reviewed"
    DECIDED = "decided"


class ReviewCyclePhase(str, Enum):
    """Phase of a review cycle."""

    SETUP = "setup"
    CALIBRATION = "calibration"
    REVIEW = "review"
    AGGREGATION = "aggregation"
    COMPLETE = "complete"


class AggregationMethod(str, Enum):
    """Method for aggregating reviewer scores."""

    WEIGHTED_AVERAGE = "weighted_average"
    TRIMMED_MEAN = "trimmed_mean"
    MEDIAN = "median"


@dataclass
class Application:
    """Represents a single application submission."""

    application_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    raw_text: str = ""
    anonymized_text: str = ""
    anonymous_id: str = ""
    metadata: dict = field(default_factory=dict)
    status: ApplicationStatus = ApplicationStatus.SUBMITTED
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = ApplicationStatus(self.status)


@dataclass
class Reviewer:
    """Represents a reviewer in the system."""

    reviewer_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    email: str = ""
    institution: str = ""
    conflicts: list[str] = field(default_factory=list)
    weight: float = 1.0
    is_calibrated: bool = False
    calibration_deviation: float = 0.0

    @property
    def display_id(self) -> str:
        """Return an anonymized display identifier for the reviewer."""
        return f"Reviewer-{self.reviewer_id[:8]}"


@dataclass
class ReviewCycle:
    """Represents a complete review cycle with its configuration."""

    cycle_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    reviews_per_application: int = 3
    aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    phase: ReviewCyclePhase = ReviewCyclePhase.SETUP
    applications: list[Application] = field(default_factory=list)
    reviewers: list[Reviewer] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    outlier_threshold_sigma: float = 2.0

    def __post_init__(self) -> None:
        if isinstance(self.phase, str):
            self.phase = ReviewCyclePhase(self.phase)
        if isinstance(self.aggregation_method, str):
            self.aggregation_method = AggregationMethod(self.aggregation_method)


@dataclass
class Review:
    """A single review: one reviewer's scores for one application."""

    review_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    application_id: str = ""
    reviewer_id: str = ""
    cycle_id: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    comments: dict[str, str] = field(default_factory=dict)
    overall_comment: str = ""
    submitted_at: Optional[datetime] = None
    is_calibration: bool = False

    @property
    def is_complete(self) -> bool:
        """Check if all score dimensions have been filled."""
        return len(self.scores) > 0 and all(
            v is not None for v in self.scores.values()
        )


@dataclass
class CalibrationResult:
    """Results of calibration analysis for a single reviewer."""

    reviewer_id: str = ""
    mean_deviation: float = 0.0
    std_deviation: float = 0.0
    dimension_deviations: dict[str, float] = field(default_factory=dict)
    is_flagged: bool = False
    flag_reason: str = ""


@dataclass
class AggregatedScore:
    """Final aggregated score for an application."""

    application_id: str = ""
    anonymous_id: str = ""
    dimension_scores: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    num_reviews: int = 0
    num_outliers_removed: int = 0
    method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    confidence: float = 0.0
    rank: int = 0
