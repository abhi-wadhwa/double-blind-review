# Double-Blind Review Platform

An anonymized evaluation platform implementing **double-blind peer review** with structured rubrics, reviewer calibration, configurable score aggregation, and inter-rater reliability analysis.

Designed for grant committees, fellowship panels, conference program committees, and any evaluation process requiring fair, unbiased assessment of applicants.

## Why Double-Blind Review?

In traditional review processes, reviewer knowledge of an applicant's name, institution, or demographics introduces [well-documented cognitive biases](https://en.wikipedia.org/wiki/Implicit_bias). Double-blind review addresses this by:

1. **Anonymizing applications** -- stripping all personally identifiable information (PII) before reviewers see the material
2. **Anonymizing reviewers** -- applicants do not know who reviewed them
3. **Structured rubrics** -- scoring along predefined dimensions reduces subjective drift
4. **Calibration rounds** -- measuring and correcting for reviewer inconsistency

## Features

| Feature | Description |
|---|---|
| **Anonymization Engine** | Regex-based PII removal: emails, phone numbers, names (with salutations), institution names, SSNs, URLs. Assigns stable anonymous IDs (`Applicant-001`, etc.) |
| **Structured Rubrics** | Admin-defined scoring dimensions with configurable scales, weights, and anchor descriptions |
| **Balanced Assignment** | Round-robin algorithm ensuring each application gets exactly *k* reviews with balanced workload and conflict-of-interest avoidance |
| **Calibration Rounds** | All reviewers score the same calibration set; engine flags high-deviation reviewers |
| **Score Aggregation** | Three methods: weighted average, trimmed mean, median. Automatic outlier removal (scores > 2 sigma from mean) |
| **Inter-Rater Reliability** | Krippendorff's alpha for nominal, ordinal, interval, and ratio data |
| **Audit Trail** | Append-only log of every platform action with timestamps |
| **Streamlit Dashboard** | Interactive UI for admin management, reviewer scoring, calibration visualization, and result export |
| **CLI** | Command-line interface for batch processing |

## Architecture

```
double-blind-review/
├── src/
│   ├── core/
│   │   ├── anonymization.py    # PII detection and redaction
│   │   ├── rubric.py           # Scoring dimension management
│   │   ├── assignment.py       # Reviewer-to-application assignment
│   │   ├── calibration.py      # Reviewer consistency analysis
│   │   ├── aggregation.py      # Score combination with outlier handling
│   │   ├── reliability.py      # Krippendorff's alpha computation
│   │   ├── audit.py            # Action logging
│   │   └── models.py           # Data models
│   ├── viz/
│   │   └── app.py              # Streamlit dashboard
│   └── cli.py                  # Command-line interface
├── tests/                      # Comprehensive test suite
├── examples/
│   └── demo.py                 # Full workflow demonstration
├── pyproject.toml
├── Dockerfile
└── Makefile
```

## Quickstart

```bash
# Clone and install
git clone https://github.com/abhi-wadhwa/double-blind-review.git
cd double-blind-review
pip install -e ".[dev]"

# Run the demo
python -m src.cli demo

# Launch the Streamlit dashboard
streamlit run src/viz/app.py

# Run tests
pytest tests/ -v
```

## How It Works

### 1. Anonymization

The engine applies a sequence of regex passes to strip PII:

```python
from src.core.anonymization import AnonymizationEngine

engine = AnonymizationEngine()
text = "Dr. Jane Smith from MIT. Contact: jane@mit.edu, 617-555-1234."
anonymized, anon_id, stats = engine.anonymize(text, "app-001")

# anonymized: "[NAME_REDACTED] from [INSTITUTION_REDACTED]. Contact: [EMAIL_REDACTED], [PHONE_REDACTED]."
# anon_id: "Applicant-001"
# stats.total: 4
```

Detected PII types:
- **Email**: `user@domain.tld` patterns
- **Phone**: US formats `(555) 123-4567`, international `+44 1234 567890`
- **Names**: Salutation + capitalized words (`Dr. John Smith`, `Prof. Maria Garcia`)
- **Institutions**: `University of X`, `X College`, `X Institute`
- **SSNs**: `123-45-6789` patterns
- **URLs**: `http://` and `https://` links

### 2. Structured Rubrics

Define scoring dimensions with scales and weights:

```python
from src.core.rubric import RubricSystem, Dimension

rubric = RubricSystem(dimensions=[
    Dimension(name="merit", label="Technical Merit", min_score=1, max_score=5, weight=2.0),
    Dimension(name="novelty", label="Novelty", min_score=1, max_score=5, weight=1.5),
    Dimension(name="clarity", label="Clarity", min_score=1, max_score=5, weight=1.0),
])
```

### 3. Balanced Assignment

The algorithm ensures:
- Each application receives exactly *k* reviews
- Reviewer workloads differ by at most 1
- No reviewer is assigned to a conflicted application

```python
from src.core.assignment import AssignmentAlgorithm

algo = AssignmentAlgorithm(reviews_per_application=3, seed=42)
plan = algo.assign(applications, reviewers)
# plan.is_balanced == True
# plan.conflicts_avoided == N
```

### 4. Calibration

During calibration, all reviewers score the same set of applications. The engine computes each reviewer's deviation from the consensus (mean):

```
deviation_i = |score_i - mean(scores)| for each dimension
```

Reviewers whose mean absolute deviation exceeds `threshold * pooled_std` are flagged for retraining or weight adjustment.

### 5. Score Aggregation

Three configurable methods:

**Weighted Average** (default):

```
score_dim = sum(w_i * s_i) / sum(w_i)
```

where `w_i` is the reviewer's calibration weight and `s_i` is their score.

**Trimmed Mean**: Drop the highest and lowest scores, then average (requires >= 3 reviews).

**Median**: The middle value, robust to outliers.

Outlier handling: Before aggregation, scores deviating more than `sigma` standard deviations from the dimension mean are excluded:

```
|s_i - mean| > sigma * std  =>  exclude s_i
```

### 6. Inter-Rater Reliability: Krippendorff's Alpha

Krippendorff's alpha generalizes agreement coefficients to handle any number of raters, missing data, and multiple measurement levels:

```
alpha = 1 - D_o / D_e
```

where:
- **D_o** = observed disagreement (within-unit variance)
- **D_e** = expected disagreement (if ratings were random)

| alpha range | Interpretation |
|---|---|
| >= 0.800 | Reliable agreement |
| 0.667 -- 0.799 | Tentative (draw cautious conclusions) |
| < 0.667 | Unreliable (re-examine process) |
| < 0.000 | Systematic disagreement |

Supported measurement levels: `nominal`, `ordinal`, `interval`, `ratio`.

```python
from src.core.reliability import ReliabilityCalculator

calc = ReliabilityCalculator(data_type="interval")
alphas = calc.compute_from_reviews(reviews)
# {"technical_merit": 0.82, "novelty": 0.71, ...}
```

## CLI Usage

```bash
# Anonymize applications
python -m src.cli anonymize --input applications.json --output anon.json

# Generate reviewer assignments
python -m src.cli assign --apps anon.json --reviewers reviewers.json -k 3

# Aggregate scores
python -m src.cli aggregate --reviews reviews.json --method weighted_average

# Compute reliability
python -m src.cli reliability --reviews reviews.json --data-type interval

# Run full demo
python -m src.cli demo
```

## Streamlit Dashboard

The interactive dashboard provides five views:

1. **Admin Dashboard** -- Create review cycles, edit rubrics, upload applications (auto-anonymized), manage reviewers, run assignment
2. **Reviewer Interface** -- View anonymized applications, score each dimension with slider controls and anchor descriptions
3. **Calibration Results** -- Per-reviewer deviation table, dimension deviation bar charts, Krippendorff's alpha metrics
4. **Results & Analytics** -- Ranked applicant table, score distribution histograms, per-dimension breakdowns, CSV/JSON export
5. **Audit Log** -- Filterable action log with entry details and export

## Docker

```bash
docker build -t double-blind-review .
docker run -p 8501:8501 double-blind-review
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Specific test module
pytest tests/test_anonymization.py -v
```

Test coverage includes:
- **Anonymization**: No PII leaks across a diverse test corpus
- **Assignment**: Exactly *k* reviews per application, no conflicts, balanced workload
- **Calibration**: Perfect agreement detection, outlier flagging
- **Aggregation**: All three methods, outlier removal, missing reviews, ties
- **Reliability**: Perfect/no agreement, multiple data types, missing data

## License

MIT
