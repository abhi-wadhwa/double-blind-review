"""Assignment algorithm for distributing applications to reviewers.

Implements balanced round-robin assignment with conflict avoidance.
Each application is assigned to exactly *k* reviewers, and reviewer
workload is kept as balanced as possible.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from src.core.audit import AuditTrail
from src.core.models import Application, Reviewer


@dataclass
class Assignment:
    """A single reviewer-to-application assignment."""

    application_id: str
    reviewer_id: str
    anonymous_id: str = ""


@dataclass
class AssignmentPlan:
    """The complete assignment plan for a review cycle."""

    assignments: list[Assignment] = field(default_factory=list)
    reviewer_loads: dict[str, int] = field(default_factory=dict)
    unassigned: list[str] = field(default_factory=list)
    conflicts_avoided: int = 0

    @property
    def total_assignments(self) -> int:
        return len(self.assignments)

    @property
    def is_balanced(self) -> bool:
        """Check if reviewer workloads differ by at most 1."""
        if not self.reviewer_loads:
            return True
        loads = list(self.reviewer_loads.values())
        return max(loads) - min(loads) <= 1

    def get_reviewers_for_application(self, app_id: str) -> list[str]:
        """Return reviewer IDs assigned to a given application."""
        return [
            a.reviewer_id
            for a in self.assignments
            if a.application_id == app_id
        ]

    def get_applications_for_reviewer(self, reviewer_id: str) -> list[str]:
        """Return application IDs assigned to a given reviewer."""
        return [
            a.application_id
            for a in self.assignments
            if a.reviewer_id == reviewer_id
        ]


class AssignmentAlgorithm:
    """Balanced round-robin assignment with conflict avoidance.

    The algorithm works as follows:

    1. Build a conflict set: reviewer R cannot review application A if
       ``A.application_id`` is in ``R.conflicts``.
    2. For each application, assign *k* eligible reviewers, choosing
       those with the lowest current workload first (greedy balancing).
    3. If fewer than *k* non-conflicted reviewers are available, the
       application is placed in an ``unassigned`` overflow list.
    """

    def __init__(
        self,
        reviews_per_application: int = 3,
        seed: Optional[int] = None,
        audit: Optional[AuditTrail] = None,
    ) -> None:
        if reviews_per_application < 1:
            raise ValueError("reviews_per_application must be >= 1")
        self.k = reviews_per_application
        self._rng = random.Random(seed)
        self._audit = audit or AuditTrail()

    def _has_conflict(self, reviewer: Reviewer, app: Application) -> bool:
        """Return True if the reviewer has a declared conflict with the app."""
        return app.application_id in reviewer.conflicts

    def assign(
        self,
        applications: list[Application],
        reviewers: list[Reviewer],
    ) -> AssignmentPlan:
        """Generate a balanced assignment plan.

        Parameters
        ----------
        applications:
            Applications to be reviewed.
        reviewers:
            Pool of available reviewers.

        Returns
        -------
        AssignmentPlan
            The resulting plan, including any unassigned applications.
        """
        if not reviewers:
            return AssignmentPlan(
                unassigned=[a.application_id for a in applications]
            )

        plan = AssignmentPlan()
        loads: dict[str, int] = {r.reviewer_id: 0 for r in reviewers}
        conflicts_avoided = 0

        # Shuffle applications to avoid ordering bias
        apps = list(applications)
        self._rng.shuffle(apps)

        for app in apps:
            # Find eligible reviewers (no conflicts)
            eligible = [
                r
                for r in reviewers
                if not self._has_conflict(r, app)
            ]

            if len(eligible) < self.k:
                # Not enough conflict-free reviewers
                conflicts_avoided += len(reviewers) - len(eligible)
                if not eligible:
                    plan.unassigned.append(app.application_id)
                    continue

            # Sort eligible by current load (ascending), break ties randomly
            self._rng.shuffle(eligible)
            eligible.sort(key=lambda r: loads[r.reviewer_id])

            selected = eligible[: self.k]
            remaining_needed = self.k - len(selected)

            for reviewer in selected:
                plan.assignments.append(
                    Assignment(
                        application_id=app.application_id,
                        reviewer_id=reviewer.reviewer_id,
                        anonymous_id=app.anonymous_id,
                    )
                )
                loads[reviewer.reviewer_id] += 1

            if remaining_needed > 0:
                # Could not assign full k reviewers, but assigned some
                plan.unassigned.append(app.application_id)
                conflicts_avoided += remaining_needed

        plan.reviewer_loads = dict(loads)
        plan.conflicts_avoided = conflicts_avoided

        self._audit.log(
            action="assign",
            entity_id="batch",
            details={
                "total_assignments": plan.total_assignments,
                "unassigned": len(plan.unassigned),
                "conflicts_avoided": conflicts_avoided,
                "balanced": plan.is_balanced,
            },
        )

        return plan

    def verify_plan(
        self,
        plan: AssignmentPlan,
        applications: list[Application],
        reviewers: list[Reviewer],
    ) -> list[str]:
        """Validate an assignment plan for correctness.

        Returns a list of error messages.  Empty means valid.
        """
        errors: list[str] = []
        app_ids = {a.application_id for a in applications}
        reviewer_ids = {r.reviewer_id for r in reviewers}
        reviewer_map = {r.reviewer_id: r for r in reviewers}
        app_map = {a.application_id: a for a in applications}

        # Check all assignments reference valid entities
        for a in plan.assignments:
            if a.application_id not in app_ids:
                errors.append(f"Unknown application: {a.application_id}")
            if a.reviewer_id not in reviewer_ids:
                errors.append(f"Unknown reviewer: {a.reviewer_id}")

        # Check each application gets exactly k reviews (unless unassigned)
        app_review_counts: dict[str, int] = defaultdict(int)
        for a in plan.assignments:
            app_review_counts[a.application_id] += 1

        for app_id in app_ids:
            if app_id in plan.unassigned:
                continue
            count = app_review_counts.get(app_id, 0)
            if count != self.k:
                errors.append(
                    f"Application {app_id} has {count} reviews, expected {self.k}"
                )

        # Check no conflict violations
        for a in plan.assignments:
            reviewer = reviewer_map.get(a.reviewer_id)
            app = app_map.get(a.application_id)
            if reviewer and app and self._has_conflict(reviewer, app):
                errors.append(
                    f"Conflict violation: reviewer {a.reviewer_id} "
                    f"assigned to application {a.application_id}"
                )

        # Check no duplicate assignments
        seen: set[tuple[str, str]] = set()
        for a in plan.assignments:
            key = (a.application_id, a.reviewer_id)
            if key in seen:
                errors.append(
                    f"Duplicate assignment: {a.reviewer_id} -> {a.application_id}"
                )
            seen.add(key)

        return errors
