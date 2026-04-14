"""Command-line interface for the double-blind review platform.

Usage:
    python -m src.cli anonymize --input data.json --output anon.json
    python -m src.cli assign --apps anon.json --reviewers reviewers.json -k 3
    python -m src.cli aggregate --reviews reviews.json --rubric rubric.json
    python -m src.cli reliability --reviews reviews.json
    python -m src.cli demo
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from src.core.aggregation import ScoreAggregator
from src.core.anonymization import AnonymizationEngine
from src.core.assignment import AssignmentAlgorithm
from src.core.audit import AuditTrail
from src.core.calibration import CalibrationEngine
from src.core.models import (
    AggregationMethod,
    Application,
    Review,
    Reviewer,
)
from src.core.reliability import ReliabilityCalculator
from src.core.rubric import Dimension, RubricSystem


def cmd_anonymize(args: argparse.Namespace) -> None:
    """Anonymize a JSON file of applications."""
    with open(args.input) as f:
        data = json.load(f)

    entries = data if isinstance(data, list) else [data]
    engine = AnonymizationEngine()

    results = []
    for entry in entries:
        text = entry.get("text", entry.get("content", ""))
        app_id = entry.get("id", str(uuid.uuid4()))
        anon_text, anon_id, stats = engine.anonymize(text, app_id)
        results.append({
            "anonymous_id": anon_id,
            "original_id": app_id,
            "anonymized_text": anon_text,
            "redactions": stats.total,
        })
        print(f"  {anon_id}: {stats.total} redactions")

    output_path = args.output or "anonymized.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAnonymized {len(results)} applications -> {output_path}")


def cmd_assign(args: argparse.Namespace) -> None:
    """Generate reviewer assignments."""
    with open(args.apps) as f:
        app_data = json.load(f)
    with open(args.reviewers) as f:
        reviewer_data = json.load(f)

    apps = [
        Application(
            application_id=a.get("id", a.get("anonymous_id", str(uuid.uuid4()))),
            anonymous_id=a.get("anonymous_id", ""),
        )
        for a in (app_data if isinstance(app_data, list) else [app_data])
    ]

    reviewers = [
        Reviewer(
            reviewer_id=r.get("id", str(uuid.uuid4())),
            name=r.get("name", ""),
            conflicts=r.get("conflicts", []),
        )
        for r in (reviewer_data if isinstance(reviewer_data, list) else [reviewer_data])
    ]

    algo = AssignmentAlgorithm(reviews_per_application=args.k)
    plan = algo.assign(apps, reviewers)

    print(f"Total assignments: {plan.total_assignments}")
    print(f"Balanced: {plan.is_balanced}")
    print(f"Unassigned: {len(plan.unassigned)}")

    # Output assignment plan
    output = {
        "assignments": [
            {"application": a.application_id, "reviewer": a.reviewer_id}
            for a in plan.assignments
        ],
        "loads": plan.reviewer_loads,
    }
    output_path = args.output or "assignments.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Assignment plan -> {output_path}")


def cmd_aggregate(args: argparse.Namespace) -> None:
    """Aggregate review scores."""
    with open(args.reviews) as f:
        review_data = json.load(f)

    if args.rubric:
        with open(args.rubric) as f:
            rubric = RubricSystem.from_dict(json.load(f))
    else:
        rubric = RubricSystem.default_rubric()

    reviews = [
        Review(
            application_id=r["application_id"],
            reviewer_id=r.get("reviewer_id", ""),
            scores=r.get("scores", {}),
        )
        for r in (review_data if isinstance(review_data, list) else [review_data])
    ]

    method = AggregationMethod(args.method) if args.method else AggregationMethod.WEIGHTED_AVERAGE
    aggregator = ScoreAggregator(rubric=rubric, method=method)
    results = aggregator.aggregate_all(reviews)

    for r in results:
        print(f"  #{r.rank} {r.anonymous_id or r.application_id[:8]}: "
              f"score={r.overall_score:.3f} confidence={r.confidence:.2f}")

    output_path = args.output or "results.json"
    output = [
        {
            "rank": r.rank,
            "application_id": r.application_id,
            "anonymous_id": r.anonymous_id,
            "overall_score": round(r.overall_score, 4),
            "dimension_scores": {k: round(v, 4) for k, v in r.dimension_scores.items()},
            "confidence": r.confidence,
            "num_reviews": r.num_reviews,
        }
        for r in results
    ]
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults -> {output_path}")


def cmd_reliability(args: argparse.Namespace) -> None:
    """Compute inter-rater reliability."""
    with open(args.reviews) as f:
        review_data = json.load(f)

    reviews = [
        Review(
            application_id=r["application_id"],
            reviewer_id=r.get("reviewer_id", ""),
            scores=r.get("scores", {}),
        )
        for r in (review_data if isinstance(review_data, list) else [review_data])
    ]

    calc = ReliabilityCalculator(data_type=args.data_type or "interval")
    alphas = calc.compute_from_reviews(reviews)
    overall = calc.compute_overall_alpha(reviews)

    print("Krippendorff's Alpha by Dimension:")
    for dim, alpha in alphas.items():
        interp = calc.interpret_alpha(alpha)
        print(f"  {dim}: {alpha:.4f} ({interp})")

    print(f"\nOverall: {overall:.4f} ({calc.interpret_alpha(overall)})")


def cmd_demo(args: argparse.Namespace) -> None:
    """Run a complete demo workflow."""
    print("=" * 60)
    print("Double-Blind Review Platform - Demo")
    print("=" * 60)

    audit = AuditTrail()

    # 1. Create rubric
    print("\n[1] Creating rubric with 3 dimensions...")
    rubric = RubricSystem(dimensions=[
        Dimension(name="quality", label="Quality", min_score=1, max_score=5, weight=2.0),
        Dimension(name="novelty", label="Novelty", min_score=1, max_score=5, weight=1.5),
        Dimension(name="clarity", label="Clarity", min_score=1, max_score=5, weight=1.0),
    ])
    for d in rubric.dimensions:
        print(f"  - {d.label}: [{d.min_score}-{d.max_score}], weight={d.weight}")

    # 2. Anonymize applications
    print("\n[2] Anonymizing 5 sample applications...")
    engine = AnonymizationEngine(audit=audit)
    sample_texts = [
        "My name is Dr. John Smith from MIT. Contact me at john@mit.edu or 617-555-1234.",
        "I am Prof. Maria Garcia at Stanford University. Email: garcia@stanford.edu.",
        "Jane Doe, Harvard College. Research on machine learning. Phone: (555) 987-6543.",
        "Dr. Robert Chen, University of California Berkeley. Visit http://robchen.com.",
        "Ms. Sarah Johnson from Yale University. SSN: 123-45-6789. Email: sj@yale.edu.",
    ]
    apps = []
    for i, text in enumerate(sample_texts):
        anon_text, anon_id, stats = engine.anonymize(text, f"app-{i}")
        apps.append(Application(
            application_id=f"app-{i}",
            raw_text=text,
            anonymized_text=anon_text,
            anonymous_id=anon_id,
        ))
        pii = engine.verify_no_pii(anon_text)
        print(f"  {anon_id}: {stats.total} redactions, PII remaining: {pii or 'none'}")

    # 3. Create reviewers and assign
    print("\n[3] Assigning 3 reviewers, k=2 reviews per application...")
    reviewers = [
        Reviewer(reviewer_id=f"rev-{i}", name=f"Reviewer {i}", weight=1.0)
        for i in range(3)
    ]
    algo = AssignmentAlgorithm(reviews_per_application=2, seed=42, audit=audit)
    plan = algo.assign(apps, reviewers)
    print(f"  Total assignments: {plan.total_assignments}")
    print(f"  Balanced: {plan.is_balanced}")
    print(f"  Loads: {plan.reviewer_loads}")

    # 4. Simulate reviews
    print("\n[4] Simulating reviews...")
    import random
    rng = random.Random(42)
    all_reviews = []
    for assignment in plan.assignments:
        scores = {
            "quality": round(rng.uniform(2, 5), 1),
            "novelty": round(rng.uniform(1, 5), 1),
            "clarity": round(rng.uniform(2, 5), 1),
        }
        review = Review(
            application_id=assignment.application_id,
            reviewer_id=assignment.reviewer_id,
            scores=scores,
        )
        all_reviews.append(review)

    # 5. Calibration
    print("\n[5] Running calibration analysis...")
    cal = CalibrationEngine(audit=audit)
    cal_results = cal.analyze(all_reviews)
    for cr in cal_results:
        flag = " [FLAGGED]" if cr.is_flagged else ""
        print(f"  {cr.reviewer_id}: deviation={cr.mean_deviation:.3f}{flag}")

    # 6. Reliability
    print("\n[6] Computing inter-rater reliability...")
    reliability = ReliabilityCalculator(audit=audit)
    alphas = reliability.compute_from_reviews(all_reviews)
    overall = reliability.compute_overall_alpha(all_reviews)
    for dim, alpha in alphas.items():
        print(f"  {dim}: alpha={alpha:.4f} ({reliability.interpret_alpha(alpha)})")
    print(f"  Overall: {overall:.4f} ({reliability.interpret_alpha(overall)})")

    # 7. Aggregation
    print("\n[7] Aggregating scores (weighted average)...")
    aggregator = ScoreAggregator(rubric=rubric, audit=audit)
    results = aggregator.aggregate_all(all_reviews, reviewers, engine.id_map)
    for r in results:
        print(f"  #{r.rank} {r.anonymous_id}: "
              f"overall={r.overall_score:.3f}, confidence={r.confidence:.2f}")

    # 8. Audit summary
    print(f"\n[8] Audit trail: {audit.count} entries")
    print(f"  Summary: {audit.summary()}")

    print("\n" + "=" * 60)
    print("Demo complete. Run 'streamlit run src/viz/app.py' for the UI.")
    print("=" * 60)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="double-blind-review",
        description="Double-blind applicant review platform",
    )
    sub = parser.add_subparsers(dest="command")

    # anonymize
    p = sub.add_parser("anonymize", help="Anonymize applications")
    p.add_argument("--input", "-i", required=True, help="Input JSON file")
    p.add_argument("--output", "-o", help="Output JSON file")

    # assign
    p = sub.add_parser("assign", help="Generate reviewer assignments")
    p.add_argument("--apps", required=True, help="Applications JSON file")
    p.add_argument("--reviewers", required=True, help="Reviewers JSON file")
    p.add_argument("-k", type=int, default=3, help="Reviews per application")
    p.add_argument("--output", "-o", help="Output JSON file")

    # aggregate
    p = sub.add_parser("aggregate", help="Aggregate review scores")
    p.add_argument("--reviews", required=True, help="Reviews JSON file")
    p.add_argument("--rubric", help="Rubric JSON file")
    p.add_argument("--method", choices=["weighted_average", "trimmed_mean", "median"])
    p.add_argument("--output", "-o", help="Output JSON file")

    # reliability
    p = sub.add_parser("reliability", help="Compute inter-rater reliability")
    p.add_argument("--reviews", required=True, help="Reviews JSON file")
    p.add_argument("--data-type", default="interval",
                   choices=["nominal", "ordinal", "interval", "ratio"])

    # demo
    sub.add_parser("demo", help="Run a complete demo workflow")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "anonymize": cmd_anonymize,
        "assign": cmd_assign,
        "aggregate": cmd_aggregate,
        "reliability": cmd_reliability,
        "demo": cmd_demo,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
