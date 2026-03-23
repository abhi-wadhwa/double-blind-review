"""Tests for the reviewer assignment algorithm."""

from collections import Counter

import pytest

from src.core.assignment import AssignmentAlgorithm, AssignmentPlan
from src.core.models import Application, Reviewer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_apps(n: int) -> list[Application]:
    """Create n test applications."""
    return [
        Application(application_id=f"app-{i}", anonymous_id=f"Applicant-{i:03d}")
        for i in range(n)
    ]


def make_reviewers(n: int, conflicts: dict[int, list[str]] | None = None) -> list[Reviewer]:
    """Create n test reviewers with optional conflicts."""
    conflicts = conflicts or {}
    return [
        Reviewer(
            reviewer_id=f"rev-{i}",
            name=f"Reviewer {i}",
            conflicts=conflicts.get(i, []),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAssignmentAlgorithm:
    """Test suite for AssignmentAlgorithm."""

    def test_each_app_gets_k_reviews(self) -> None:
        """Every application must receive exactly k reviews."""
        apps = make_apps(10)
        reviewers = make_reviewers(5)
        algo = AssignmentAlgorithm(reviews_per_application=3, seed=42)
        plan = algo.assign(apps, reviewers)

        counts = Counter(a.application_id for a in plan.assignments)
        for app in apps:
            assert counts[app.application_id] == 3, (
                f"{app.application_id} has {counts[app.application_id]} reviews"
            )

    def test_balanced_workload(self) -> None:
        """Reviewer workloads should differ by at most 1."""
        apps = make_apps(12)
        reviewers = make_reviewers(4)
        algo = AssignmentAlgorithm(reviews_per_application=3, seed=42)
        plan = algo.assign(apps, reviewers)

        loads = list(plan.reviewer_loads.values())
        assert max(loads) - min(loads) <= 1
        assert plan.is_balanced

    def test_conflict_avoidance(self) -> None:
        """Reviewer must not be assigned to conflicted applications."""
        apps = make_apps(5)
        # Reviewer 0 has conflict with app-0 and app-1
        reviewers = make_reviewers(3, conflicts={0: ["app-0", "app-1"]})
        algo = AssignmentAlgorithm(reviews_per_application=2, seed=42)
        plan = algo.assign(apps, reviewers)

        for a in plan.assignments:
            if a.reviewer_id == "rev-0":
                assert a.application_id not in ["app-0", "app-1"], (
                    f"Conflict violation: rev-0 assigned to {a.application_id}"
                )

    def test_no_duplicate_assignments(self) -> None:
        """Same reviewer should not be assigned to same app twice."""
        apps = make_apps(8)
        reviewers = make_reviewers(4)
        algo = AssignmentAlgorithm(reviews_per_application=3, seed=42)
        plan = algo.assign(apps, reviewers)

        seen = set()
        for a in plan.assignments:
            key = (a.application_id, a.reviewer_id)
            assert key not in seen, f"Duplicate: {key}"
            seen.add(key)

    def test_insufficient_reviewers(self) -> None:
        """When fewer reviewers than k, apps go to unassigned."""
        apps = make_apps(5)
        reviewers = make_reviewers(2)
        algo = AssignmentAlgorithm(reviews_per_application=5, seed=42)
        plan = algo.assign(apps, reviewers)

        # With only 2 reviewers and k=5, all apps should be partially unassigned
        assert len(plan.unassigned) == len(apps)

    def test_no_reviewers(self) -> None:
        """With zero reviewers, all apps are unassigned."""
        apps = make_apps(3)
        algo = AssignmentAlgorithm(reviews_per_application=2, seed=42)
        plan = algo.assign(apps, [])
        assert len(plan.unassigned) == 3
        assert plan.total_assignments == 0

    def test_no_applications(self) -> None:
        """With zero applications, plan is empty."""
        reviewers = make_reviewers(3)
        algo = AssignmentAlgorithm(reviews_per_application=2, seed=42)
        plan = algo.assign([], reviewers)
        assert plan.total_assignments == 0
        assert len(plan.unassigned) == 0

    def test_k_equals_one(self) -> None:
        """k=1 should assign exactly one reviewer per app."""
        apps = make_apps(6)
        reviewers = make_reviewers(3)
        algo = AssignmentAlgorithm(reviews_per_application=1, seed=42)
        plan = algo.assign(apps, reviewers)

        counts = Counter(a.application_id for a in plan.assignments)
        for app in apps:
            assert counts[app.application_id] == 1

    def test_verify_plan_valid(self) -> None:
        """verify_plan should return no errors for a correct plan."""
        apps = make_apps(6)
        reviewers = make_reviewers(4)
        algo = AssignmentAlgorithm(reviews_per_application=2, seed=42)
        plan = algo.assign(apps, reviewers)
        errors = algo.verify_plan(plan, apps, reviewers)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_verify_plan_detects_conflict(self) -> None:
        """verify_plan should detect conflict violations in a tampered plan."""
        apps = make_apps(3)
        reviewers = make_reviewers(3, conflicts={0: ["app-0"]})
        algo = AssignmentAlgorithm(reviews_per_application=2, seed=42)
        plan = algo.assign(apps, reviewers)

        # Tamper: force rev-0 onto app-0
        from src.core.assignment import Assignment
        plan.assignments.append(Assignment(
            application_id="app-0", reviewer_id="rev-0"
        ))

        errors = algo.verify_plan(plan, apps, reviewers)
        assert any("Conflict violation" in e for e in errors)

    def test_invalid_k(self) -> None:
        """k < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            AssignmentAlgorithm(reviews_per_application=0)

    def test_plan_helpers(self) -> None:
        """Test AssignmentPlan helper methods."""
        apps = make_apps(4)
        reviewers = make_reviewers(3)
        algo = AssignmentAlgorithm(reviews_per_application=2, seed=42)
        plan = algo.assign(apps, reviewers)

        for app in apps:
            assigned = plan.get_reviewers_for_application(app.application_id)
            assert len(assigned) == 2

        for rev in reviewers:
            assigned = plan.get_applications_for_reviewer(rev.reviewer_id)
            assert len(assigned) >= 1

    def test_large_scale(self) -> None:
        """Stress test with 100 applications and 20 reviewers."""
        apps = make_apps(100)
        reviewers = make_reviewers(20)
        algo = AssignmentAlgorithm(reviews_per_application=3, seed=42)
        plan = algo.assign(apps, reviewers)

        counts = Counter(a.application_id for a in plan.assignments)
        for app in apps:
            assert counts[app.application_id] == 3

        loads = list(plan.reviewer_loads.values())
        assert max(loads) - min(loads) <= 1
