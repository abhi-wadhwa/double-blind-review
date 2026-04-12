"""Streamlit application for the double-blind review platform.

Run with:
    streamlit run src/viz/app.py

Provides four main views:
1. Admin Dashboard - Create review cycles, define rubrics, upload applications
2. Reviewer Interface - Score anonymized applications
3. Calibration Results - Reviewer agreement visualization
4. Results & Analytics - Score distributions, reliability, rankings
"""

from __future__ import annotations

import io
import json
import csv
import uuid
from typing import Any

import streamlit as st

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
    ReviewCycle,
    Review,
    AggregationMethod,
    ReviewCyclePhase,
)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------


def _init_state() -> None:
    """Initialize Streamlit session state with default values."""
    defaults: dict[str, Any] = {
        "audit": AuditTrail(actor="streamlit-ui"),
        "cycles": {},           # cycle_id -> ReviewCycle
        "rubrics": {},          # cycle_id -> RubricSystem
        "applications": {},     # cycle_id -> list[Application]
        "reviewers": {},        # cycle_id -> list[Reviewer]
        "reviews": {},          # cycle_id -> list[Review]
        "anon_engines": {},     # cycle_id -> AnonymizationEngine
        "assignment_plans": {}, # cycle_id -> AssignmentPlan
        "active_cycle": None,
        "active_reviewer": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------


def _sidebar() -> str:
    """Render sidebar and return the selected page."""
    st.sidebar.title("Double-Blind Review")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Admin Dashboard",
            "Reviewer Interface",
            "Calibration Results",
            "Results & Analytics",
            "Audit Log",
        ],
    )

    # Cycle selector
    cycles = st.session_state["cycles"]
    if cycles:
        st.sidebar.markdown("---")
        cycle_names = {cid: c.name for cid, c in cycles.items()}
        selected = st.sidebar.selectbox(
            "Active Cycle",
            list(cycle_names.keys()),
            format_func=lambda x: cycle_names[x],
        )
        st.session_state["active_cycle"] = selected
    else:
        st.session_state["active_cycle"] = None

    return page


# ---------------------------------------------------------------------------
# Page: Admin Dashboard
# ---------------------------------------------------------------------------


def _page_admin() -> None:
    """Admin dashboard for managing review cycles."""
    st.header("Admin Dashboard")

    # --- Create new cycle ---
    with st.expander("Create New Review Cycle", expanded=not st.session_state["cycles"]):
        with st.form("create_cycle"):
            name = st.text_input("Cycle Name", "Fellowship 2025")
            desc = st.text_area("Description", "Annual fellowship application review")
            k = st.number_input("Reviews per Application", 2, 10, 3)
            method = st.selectbox(
                "Aggregation Method",
                [m.value for m in AggregationMethod],
            )
            submitted = st.form_submit_button("Create Cycle")

        if submitted and name:
            cycle = ReviewCycle(
                name=name,
                description=desc,
                reviews_per_application=k,
                aggregation_method=AggregationMethod(method),
            )
            cid = cycle.cycle_id
            st.session_state["cycles"][cid] = cycle
            st.session_state["rubrics"][cid] = RubricSystem.default_rubric()
            st.session_state["applications"][cid] = []
            st.session_state["reviewers"][cid] = []
            st.session_state["reviews"][cid] = []
            st.session_state["anon_engines"][cid] = AnonymizationEngine(
                audit=st.session_state["audit"]
            )
            st.session_state["active_cycle"] = cid
            st.session_state["audit"].log("create_cycle", cid, {"name": name})
            st.success(f"Created cycle: {name}")
            st.rerun()

    cid = st.session_state["active_cycle"]
    if not cid:
        st.info("Create a review cycle to get started.")
        return

    cycle = st.session_state["cycles"][cid]
    st.subheader(f"Cycle: {cycle.name}")
    st.caption(f"Phase: {cycle.phase.value} | ID: {cid[:8]}...")

    # --- Rubric editor ---
    with st.expander("Edit Rubric"):
        rubric = st.session_state["rubrics"][cid]
        for dim in rubric.dimensions:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.text(f"{dim.label} ({dim.name})")
            col2.text(f"[{dim.min_score}-{dim.max_score}]")
            col3.text(f"w={dim.weight}")

        st.markdown("**Add Dimension**")
        with st.form("add_dim"):
            dname = st.text_input("Name (snake_case)")
            dlabel = st.text_input("Label")
            dmin = st.number_input("Min Score", value=1.0)
            dmax = st.number_input("Max Score", value=5.0)
            dweight = st.number_input("Weight", value=1.0, min_value=0.0)
            if st.form_submit_button("Add Dimension") and dname:
                try:
                    rubric.add_dimension(Dimension(
                        name=dname, label=dlabel,
                        min_score=dmin, max_score=dmax, weight=dweight,
                    ))
                    st.success(f"Added dimension: {dlabel or dname}")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

    # --- Upload applications ---
    with st.expander("Upload Applications"):
        uploaded = st.file_uploader(
            "Upload CSV or JSON",
            type=["csv", "json"],
            key=f"upload_{cid}",
        )
        if uploaded:
            engine = st.session_state["anon_engines"][cid]
            apps = st.session_state["applications"][cid]

            if uploaded.name.endswith(".json"):
                data = json.load(uploaded)
                entries = data if isinstance(data, list) else [data]
            else:
                reader = csv.DictReader(io.StringIO(uploaded.read().decode()))
                entries = list(reader)

            new_count = 0
            for entry in entries:
                text = entry.get("text", entry.get("content", entry.get("body", "")))
                app_id = entry.get("id", str(uuid.uuid4()))
                anon_text, anon_id, stats = engine.anonymize(text, app_id)
                app = Application(
                    application_id=app_id,
                    raw_text=text,
                    anonymized_text=anon_text,
                    anonymous_id=anon_id,
                )
                apps.append(app)
                new_count += 1

            st.success(f"Uploaded and anonymized {new_count} applications.")
            st.rerun()

        st.write(f"**Total applications:** {len(st.session_state['applications'].get(cid, []))}")

    # --- Manage reviewers ---
    with st.expander("Manage Reviewers"):
        reviewers = st.session_state["reviewers"][cid]
        for r in reviewers:
            st.text(f"{r.name} ({r.email}) - weight: {r.weight}")

        with st.form("add_reviewer"):
            rname = st.text_input("Name")
            remail = st.text_input("Email")
            rinst = st.text_input("Institution")
            rweight = st.number_input("Weight", value=1.0, min_value=0.0)
            if st.form_submit_button("Add Reviewer") and rname:
                reviewer = Reviewer(
                    name=rname, email=remail,
                    institution=rinst, weight=rweight,
                )
                reviewers.append(reviewer)
                st.success(f"Added reviewer: {rname}")
                st.rerun()

    # --- Run assignment ---
    apps = st.session_state["applications"].get(cid, [])
    reviewers = st.session_state["reviewers"].get(cid, [])
    if apps and reviewers:
        if st.button("Run Reviewer Assignment"):
            algo = AssignmentAlgorithm(
                reviews_per_application=cycle.reviews_per_application,
                audit=st.session_state["audit"],
            )
            plan = algo.assign(apps, reviewers)
            st.session_state["assignment_plans"][cid] = plan
            cycle.phase = ReviewCyclePhase.REVIEW
            st.success(
                f"Assigned {plan.total_assignments} reviews. "
                f"Balanced: {plan.is_balanced}. "
                f"Unassigned: {len(plan.unassigned)}."
            )
            st.rerun()


# ---------------------------------------------------------------------------
# Page: Reviewer Interface
# ---------------------------------------------------------------------------


def _page_reviewer() -> None:
    """Interface for reviewers to score anonymized applications."""
    st.header("Reviewer Interface")

    cid = st.session_state["active_cycle"]
    if not cid:
        st.info("No active review cycle.")
        return

    reviewers = st.session_state["reviewers"].get(cid, [])
    if not reviewers:
        st.info("No reviewers registered for this cycle.")
        return

    plan = st.session_state.get("assignment_plans", {}).get(cid)
    if not plan:
        st.info("Assignment has not been run yet.")
        return

    # Reviewer selection
    reviewer_names = {r.reviewer_id: r.name for r in reviewers}
    selected_rid = st.selectbox(
        "Select Reviewer",
        list(reviewer_names.keys()),
        format_func=lambda x: reviewer_names[x],
    )
    st.session_state["active_reviewer"] = selected_rid

    # Get assigned applications
    assigned_app_ids = plan.get_applications_for_reviewer(selected_rid)
    apps = st.session_state["applications"].get(cid, [])
    app_map = {a.application_id: a for a in apps}
    rubric = st.session_state["rubrics"][cid]

    existing_reviews = st.session_state["reviews"].get(cid, [])
    reviewed_ids = {
        r.application_id
        for r in existing_reviews
        if r.reviewer_id == selected_rid
    }

    st.write(f"**Assigned:** {len(assigned_app_ids)} applications | "
             f"**Completed:** {len(reviewed_ids)}")

    for app_id in assigned_app_ids:
        app = app_map.get(app_id)
        if not app:
            continue

        status = "Reviewed" if app_id in reviewed_ids else "Pending"
        with st.expander(f"{app.anonymous_id} [{status}]"):
            st.markdown("**Application Text (Anonymized):**")
            st.text_area(
                "Content",
                app.anonymized_text,
                height=200,
                disabled=True,
                key=f"text_{app_id}_{selected_rid}",
            )

            if app_id not in reviewed_ids:
                with st.form(f"review_{app_id}_{selected_rid}"):
                    scores: dict[str, float] = {}
                    comments: dict[str, str] = {}
                    for dim in rubric.dimensions:
                        scores[dim.name] = st.slider(
                            dim.label,
                            min_value=float(dim.min_score),
                            max_value=float(dim.max_score),
                            value=float((dim.min_score + dim.max_score) / 2),
                            step=0.5,
                            key=f"score_{app_id}_{selected_rid}_{dim.name}",
                            help=dim.description,
                        )
                        if dim.anchors:
                            anchor_text = " | ".join(
                                f"{k}: {v}" for k, v in sorted(dim.anchors.items())
                            )
                            st.caption(anchor_text)
                        comments[dim.name] = st.text_input(
                            f"Comment on {dim.label}",
                            key=f"comment_{app_id}_{selected_rid}_{dim.name}",
                        )

                    overall_comment = st.text_area(
                        "Overall Comments",
                        key=f"overall_{app_id}_{selected_rid}",
                    )

                    if st.form_submit_button("Submit Review"):
                        review = Review(
                            application_id=app_id,
                            reviewer_id=selected_rid,
                            cycle_id=cid,
                            scores=scores,
                            comments=comments,
                            overall_comment=overall_comment,
                        )
                        st.session_state["reviews"][cid].append(review)
                        st.session_state["audit"].log(
                            "submit_review", app_id,
                            {"reviewer": selected_rid},
                        )
                        st.success(f"Review submitted for {app.anonymous_id}")
                        st.rerun()


# ---------------------------------------------------------------------------
# Page: Calibration Results
# ---------------------------------------------------------------------------


def _page_calibration() -> None:
    """Show calibration analysis and reviewer agreement."""
    st.header("Calibration Results")

    cid = st.session_state["active_cycle"]
    if not cid:
        st.info("No active review cycle.")
        return

    reviews = st.session_state["reviews"].get(cid, [])
    if not reviews:
        st.info("No reviews submitted yet.")
        return

    # Run calibration
    cal_engine = CalibrationEngine(audit=st.session_state["audit"])
    cal_results = cal_engine.analyze(reviews)

    if not cal_results:
        st.info("Not enough data for calibration analysis.")
        return

    reviewer_map = {
        r.reviewer_id: r.name
        for r in st.session_state["reviewers"].get(cid, [])
    }

    # Summary table
    st.subheader("Reviewer Consistency")
    table_data = []
    for cr in cal_results:
        table_data.append({
            "Reviewer": reviewer_map.get(cr.reviewer_id, cr.reviewer_id[:8]),
            "Mean Deviation": round(cr.mean_deviation, 3),
            "Std Deviation": round(cr.std_deviation, 3),
            "Flagged": "Yes" if cr.is_flagged else "No",
            "Reason": cr.flag_reason if cr.is_flagged else "-",
        })
    st.table(table_data)

    # Per-dimension deviation chart
    st.subheader("Deviation by Dimension")
    chart_data: dict[str, list[float]] = {}
    labels: list[str] = []
    for cr in cal_results:
        name = reviewer_map.get(cr.reviewer_id, cr.reviewer_id[:8])
        labels.append(name)
        for dim, dev in cr.dimension_deviations.items():
            chart_data.setdefault(dim, []).append(dev)

    if chart_data:
        import pandas as pd
        df = pd.DataFrame(chart_data, index=labels)
        st.bar_chart(df)

    # Inter-rater reliability
    st.subheader("Inter-Rater Reliability (Krippendorff's Alpha)")
    reliability = ReliabilityCalculator(audit=st.session_state["audit"])
    alphas = reliability.compute_from_reviews(reviews)
    overall_alpha = reliability.compute_overall_alpha(reviews)

    for dim, alpha in alphas.items():
        interp = reliability.interpret_alpha(alpha)
        st.metric(dim, f"{alpha:.3f}", help=interp)

    st.metric("Overall Alpha", f"{overall_alpha:.3f}",
              help=reliability.interpret_alpha(overall_alpha))


# ---------------------------------------------------------------------------
# Page: Results & Analytics
# ---------------------------------------------------------------------------


def _page_results() -> None:
    """Show aggregated results, score distributions, and rankings."""
    st.header("Results & Analytics")

    cid = st.session_state["active_cycle"]
    if not cid:
        st.info("No active review cycle.")
        return

    reviews = st.session_state["reviews"].get(cid, [])
    if not reviews:
        st.info("No reviews submitted yet.")
        return

    cycle = st.session_state["cycles"][cid]
    rubric = st.session_state["rubrics"][cid]
    reviewers = st.session_state["reviewers"].get(cid, [])
    engine = st.session_state["anon_engines"].get(cid)
    anon_map = engine.id_map if engine else {}

    # Aggregate
    aggregator = ScoreAggregator(
        rubric=rubric,
        method=cycle.aggregation_method,
        outlier_sigma=cycle.outlier_threshold_sigma,
        audit=st.session_state["audit"],
    )
    results = aggregator.aggregate_all(reviews, reviewers, anon_map)

    # Rankings table
    st.subheader("Rankings")
    import pandas as pd

    ranking_data = []
    for r in results:
        row = {
            "Rank": r.rank,
            "Applicant": r.anonymous_id or r.application_id[:8],
            "Overall Score": round(r.overall_score, 3),
            "Reviews": r.num_reviews,
            "Confidence": round(r.confidence, 2),
            "Outliers Removed": r.num_outliers_removed,
        }
        for dim, score in r.dimension_scores.items():
            row[dim] = round(score, 2)
        ranking_data.append(row)

    df = pd.DataFrame(ranking_data)
    st.dataframe(df, use_container_width=True)

    # Score distribution histogram
    st.subheader("Score Distribution")
    overall_scores = [r.overall_score for r in results]
    if overall_scores:
        hist_df = pd.DataFrame({"Overall Score": overall_scores})
        st.bar_chart(hist_df["Overall Score"].value_counts().sort_index())

    # Dimension-level distributions
    st.subheader("Dimension Score Distributions")
    for dim in rubric.dimensions:
        dim_vals = [
            r.dimension_scores.get(dim.name, 0) for r in results
            if dim.name in r.dimension_scores
        ]
        if dim_vals:
            st.write(f"**{dim.label}**")
            dim_df = pd.DataFrame({dim.label: dim_vals})
            st.bar_chart(dim_df[dim.label].value_counts().sort_index())

    # Export
    st.subheader("Export Results")
    csv_buffer = io.StringIO()
    if ranking_data:
        writer = csv.DictWriter(csv_buffer, fieldnames=ranking_data[0].keys())
        writer.writeheader()
        writer.writerows(ranking_data)
        st.download_button(
            "Download CSV",
            csv_buffer.getvalue(),
            file_name="review_results.csv",
            mime="text/csv",
        )

    json_export = json.dumps(ranking_data, indent=2, default=str)
    st.download_button(
        "Download JSON",
        json_export,
        file_name="review_results.json",
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# Page: Audit Log
# ---------------------------------------------------------------------------


def _page_audit() -> None:
    """Display the audit trail."""
    st.header("Audit Log")

    audit: AuditTrail = st.session_state["audit"]

    if audit.count == 0:
        st.info("No audit entries yet.")
        return

    # Summary
    st.subheader("Summary")
    summary = audit.summary()
    cols = st.columns(len(summary))
    for col, (action, count) in zip(cols, summary.items()):
        col.metric(action, count)

    # Filter
    st.subheader("Entries")
    action_filter = st.selectbox(
        "Filter by action",
        ["All"] + list(summary.keys()),
    )

    entries = audit.entries
    if action_filter != "All":
        entries = [e for e in entries if e.action == action_filter]

    for entry in reversed(entries[-100:]):
        with st.expander(
            f"{entry.timestamp} | {entry.action} | {entry.entity_id}"
        ):
            st.json(entry.details)

    # Export
    st.download_button(
        "Export Audit Log (JSON)",
        audit.export_json(),
        file_name="audit_log.json",
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(
        page_title="Double-Blind Review Platform",
        page_icon="",
        layout="wide",
    )

    _init_state()
    page = _sidebar()

    if page == "Admin Dashboard":
        _page_admin()
    elif page == "Reviewer Interface":
        _page_reviewer()
    elif page == "Calibration Results":
        _page_calibration()
    elif page == "Results & Analytics":
        _page_results()
    elif page == "Audit Log":
        _page_audit()


if __name__ == "__main__":
    main()
