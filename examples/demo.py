"""Complete demonstration of the double-blind review platform.

Run:
    python -m examples.demo

This script walks through the full workflow:
1. Define a scoring rubric
2. Upload and anonymize applications
3. Assign reviewers
4. Simulate calibration and review scoring
5. Analyze calibration results
6. Aggregate scores
7. Compute inter-rater reliability
8. Display ranked results
"""

from __future__ import annotations

import random

from src.core.anonymization import AnonymizationEngine
from src.core.rubric import RubricSystem, Dimension
from src.core.assignment import AssignmentAlgorithm
from src.core.calibration import CalibrationEngine
from src.core.aggregation import ScoreAggregator
from src.core.reliability import ReliabilityCalculator
from src.core.audit import AuditTrail
from src.core.models import (
    Application,
    Reviewer,
    Review,
    AggregationMethod,
)


def main() -> None:
    """Run the complete demo workflow."""
    rng = random.Random(42)
    audit = AuditTrail(actor="demo")

    # ---------------------------------------------------------------
    # 1. Define Rubric
    # ---------------------------------------------------------------
    print("=" * 70)
    print("DOUBLE-BLIND REVIEW PLATFORM - FULL DEMO")
    print("=" * 70)

    print("\n--- Step 1: Define Scoring Rubric ---")
    rubric = RubricSystem(
        dimensions=[
            Dimension(
                name="technical_merit",
                label="Technical Merit",
                description="Rigor and correctness of approach",
                min_score=1,
                max_score=5,
                weight=2.0,
                anchors={1: "Weak", 3: "Competent", 5: "Outstanding"},
            ),
            Dimension(
                name="innovation",
                label="Innovation",
                description="Novelty and creative contribution",
                min_score=1,
                max_score=5,
                weight=1.5,
            ),
            Dimension(
                name="presentation",
                label="Presentation Quality",
                description="Clarity, structure, writing quality",
                min_score=1,
                max_score=5,
                weight=1.0,
            ),
            Dimension(
                name="impact",
                label="Potential Impact",
                description="Broader significance of the work",
                min_score=1,
                max_score=5,
                weight=1.5,
            ),
        ]
    )

    for dim in rubric.dimensions:
        print(f"  {dim.label}: [{dim.min_score}-{dim.max_score}], weight={dim.weight}")

    # ---------------------------------------------------------------
    # 2. Upload & Anonymize Applications
    # ---------------------------------------------------------------
    print("\n--- Step 2: Upload & Anonymize Applications ---")

    raw_applications = [
        {
            "id": "APP-2025-001",
            "text": (
                "My name is Dr. Alice Thompson from the University of Cambridge. "
                "I propose a novel approach to quantum error correction using "
                "topological codes. Contact: alice.thompson@cam.ac.uk, +44 1234 567890. "
                "See my work at https://alice-qec.github.io."
            ),
        },
        {
            "id": "APP-2025-002",
            "text": (
                "Prof. Rajesh Patel, Indian Institute of Technology Mumbai. "
                "Our team has developed a scalable framework for federated learning "
                "in healthcare. Email: r.patel@iitm.ac.in. Phone: 555-222-3333."
            ),
        },
        {
            "id": "APP-2025-003",
            "text": (
                "I am Ms. Clara Rodriguez, Stanford University. This proposal "
                "introduces adaptive curriculum learning for low-resource NLP. "
                "Email: clara@stanford.edu."
            ),
        },
        {
            "id": "APP-2025-004",
            "text": (
                "Dr. Michael Weber, Max Planck Institute for Computer Science. "
                "We present a new algorithm for graph neural networks that achieves "
                "O(n log n) complexity. mweber@mpi-cs.mpg.de."
            ),
        },
        {
            "id": "APP-2025-005",
            "text": (
                "Submitted by Dr. Yuki Tanaka from the University of Tokyo. "
                "This work explores bio-inspired optimization methods for "
                "chip design. ytanaka@u-tokyo.ac.jp, http://tanaka-lab.jp."
            ),
        },
        {
            "id": "APP-2025-006",
            "text": (
                "Mr. James O'Brien, MIT. A framework for privacy-preserving "
                "machine learning using secure multi-party computation. "
                "james.obrien@mit.edu, 617-555-8888."
            ),
        },
        {
            "id": "APP-2025-007",
            "text": (
                "Prof. Amara Okafor from University of Lagos. Our research "
                "explores AI-driven early warning systems for tropical diseases. "
                "a.okafor@unilag.edu.ng, +234 801 555 6789."
            ),
        },
        {
            "id": "APP-2025-008",
            "text": (
                "Dr. Sophie Muller, ETH Zurich. This proposal develops "
                "interpretable deep learning for medical imaging diagnostics. "
                "sophie.muller@ethz.ch, www.ethz-ml.ch/sophie."
            ),
        },
    ]

    engine = AnonymizationEngine(audit=audit)
    applications: list[Application] = []

    for raw in raw_applications:
        anon_text, anon_id, stats = engine.anonymize(raw["text"], raw["id"])
        app = Application(
            application_id=raw["id"],
            raw_text=raw["text"],
            anonymized_text=anon_text,
            anonymous_id=anon_id,
        )
        applications.append(app)

        remaining_pii = engine.verify_no_pii(anon_text)
        status = "CLEAN" if not remaining_pii else f"WARNING: {remaining_pii}"
        print(f"  {anon_id} ({raw['id']}): {stats.total} redactions [{status}]")

    # ---------------------------------------------------------------
    # 3. Register Reviewers & Assign
    # ---------------------------------------------------------------
    print("\n--- Step 3: Register Reviewers & Assign ---")

    reviewers = [
        Reviewer(reviewer_id="R1", name="Expert A", weight=1.2, conflicts=["APP-2025-001"]),
        Reviewer(reviewer_id="R2", name="Expert B", weight=1.0),
        Reviewer(reviewer_id="R3", name="Expert C", weight=1.0, conflicts=["APP-2025-003"]),
        Reviewer(reviewer_id="R4", name="Expert D", weight=0.8),
        Reviewer(reviewer_id="R5", name="Expert E", weight=1.1),
    ]

    k = 3
    algo = AssignmentAlgorithm(reviews_per_application=k, seed=42, audit=audit)
    plan = algo.assign(applications, reviewers)

    print(f"  Reviews per application: {k}")
    print(f"  Total assignments: {plan.total_assignments}")
    print(f"  Balanced workload: {plan.is_balanced}")
    print(f"  Reviewer loads: {plan.reviewer_loads}")
    print(f"  Conflicts avoided: {plan.conflicts_avoided}")
    print(f"  Unassigned: {len(plan.unassigned)}")

    # Verify plan
    errors = algo.verify_plan(plan, applications, reviewers)
    if errors:
        print(f"  PLAN ERRORS: {errors}")
    else:
        print("  Plan verified: no errors")

    # ---------------------------------------------------------------
    # 4. Simulate Calibration Round
    # ---------------------------------------------------------------
    print("\n--- Step 4: Calibration Round ---")

    # Use first 2 applications as calibration items
    cal_apps = applications[:2]
    cal_reviews: list[Review] = []

    # Simulate: most reviewers agree, one is an outlier
    for app in cal_apps:
        consensus = {
            "technical_merit": rng.uniform(3.0, 4.5),
            "innovation": rng.uniform(3.0, 4.5),
            "presentation": rng.uniform(3.0, 4.0),
            "impact": rng.uniform(3.0, 4.5),
        }
        for reviewer in reviewers:
            if reviewer.reviewer_id == "R4":
                # Outlier reviewer - always scores low
                scores = {dim: max(1.0, consensus[dim] - 2.0) for dim in consensus}
            else:
                # Normal reviewer with small variance
                scores = {
                    dim: round(min(5.0, max(1.0, val + rng.gauss(0, 0.3))), 1)
                    for dim, val in consensus.items()
                }
            cal_reviews.append(
                Review(
                    application_id=app.application_id,
                    reviewer_id=reviewer.reviewer_id,
                    scores=scores,
                    is_calibration=True,
                )
            )

    cal_engine = CalibrationEngine(threshold=1.5, audit=audit)
    cal_results = cal_engine.analyze(cal_reviews)

    for cr in cal_results:
        name = next(r.name for r in reviewers if r.reviewer_id == cr.reviewer_id)
        flag = " ** FLAGGED **" if cr.is_flagged else ""
        print(f"  {name} ({cr.reviewer_id}): "
              f"mean_dev={cr.mean_deviation:.3f}, "
              f"std_dev={cr.std_deviation:.3f}{flag}")
        if cr.is_flagged:
            print(f"    Reason: {cr.flag_reason}")

    # ---------------------------------------------------------------
    # 5. Simulate Main Review Round
    # ---------------------------------------------------------------
    print("\n--- Step 5: Main Review Round ---")

    all_reviews: list[Review] = []

    for assignment in plan.assignments:
        # Generate plausible scores with some reviewer-specific bias
        base_quality = rng.uniform(2.5, 4.5)
        reviewer_bias = {"R1": 0.3, "R2": 0.0, "R3": -0.2, "R4": -0.8, "R5": 0.1}
        bias = reviewer_bias.get(assignment.reviewer_id, 0.0)

        scores = {
            "technical_merit": round(min(5, max(1, base_quality + bias + rng.gauss(0, 0.4))), 1),
            "innovation": round(min(5, max(1, base_quality - 0.5 + bias + rng.gauss(0, 0.5))), 1),
            "presentation": round(min(5, max(1, base_quality + 0.3 + rng.gauss(0, 0.3))), 1),
            "impact": round(min(5, max(1, base_quality + bias + rng.gauss(0, 0.4))), 1),
        }

        review = Review(
            application_id=assignment.application_id,
            reviewer_id=assignment.reviewer_id,
            scores=scores,
        )
        all_reviews.append(review)

    print(f"  Total reviews submitted: {len(all_reviews)}")

    # ---------------------------------------------------------------
    # 6. Inter-Rater Reliability
    # ---------------------------------------------------------------
    print("\n--- Step 6: Inter-Rater Reliability ---")

    reliability = ReliabilityCalculator(data_type="interval", audit=audit)
    alphas = reliability.compute_from_reviews(all_reviews)
    overall_alpha = reliability.compute_overall_alpha(all_reviews)

    for dim, alpha in sorted(alphas.items()):
        interp = reliability.interpret_alpha(alpha)
        print(f"  {dim}: alpha = {alpha:.4f} ({interp})")
    print(f"  Overall: alpha = {overall_alpha:.4f} "
          f"({reliability.interpret_alpha(overall_alpha)})")

    # ---------------------------------------------------------------
    # 7. Aggregate Scores
    # ---------------------------------------------------------------
    print("\n--- Step 7: Score Aggregation ---")

    # Try multiple methods
    for method in AggregationMethod:
        aggregator = ScoreAggregator(
            rubric=rubric,
            method=method,
            outlier_sigma=2.0,
            audit=audit,
        )
        results = aggregator.aggregate_all(
            all_reviews, reviewers, engine.id_map
        )

        print(f"\n  Method: {method.value}")
        print(f"  {'Rank':<6} {'Applicant':<16} {'Overall':>8} {'Confidence':>11} {'Outliers':>9}")
        print(f"  {'-' * 52}")
        for r in results:
            print(
                f"  {r.rank:<6} {r.anonymous_id:<16} "
                f"{r.overall_score:>8.3f} {r.confidence:>11.2f} "
                f"{r.num_outliers_removed:>9}"
            )

    # ---------------------------------------------------------------
    # 8. Audit Summary
    # ---------------------------------------------------------------
    print("\n--- Step 8: Audit Trail Summary ---")
    print(f"  Total entries: {audit.count}")
    for action, count in audit.summary().items():
        print(f"    {action}: {count}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print(f"  - {len(applications)} applications anonymized")
    print(f"  - {len(reviewers)} reviewers assigned")
    print(f"  - {len(all_reviews)} reviews processed")
    print(f"  - Inter-rater reliability computed (Krippendorff's alpha)")
    print(f"  - Scores aggregated and ranked")
    print("=" * 70)


if __name__ == "__main__":
    main()
