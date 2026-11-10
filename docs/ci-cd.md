# Continuous Integration (CI) Automation Architecture

## 1. What Problem Does CI Solve in Machine Learning?

In traditional software development, CI compiles code and runs unit tests. In Machine Learning, breaking changes can be much more elusive:
- A data scientist changes a feature name in preprocessing, breaking the downstream prediction API.
- Someone commits code with syntax errors or formatting violations.
- A library update changes the serialization format of a model.
- A Dockerfile is modified in a way that breaks container initialization.

**Continuous Integration (CI)** runs automatically on every pull request and push to ensure that:
1. All code adheres to standardized formatting and linting rules.
2. Unit tests, schema validations, and drift detectors pass.
3. The end-to-end data pipeline (`generate -> validate -> preprocess -> train`) runs to completion without errors.
4. The production Docker container compiles cleanly without broken dependencies.

---

## 2. GitHub Actions CI Workflow Structure

Our CI workflow (`.github/workflows/ci.yml`) is split into two sequential jobs:

```mermaid
flowchart TD
    Push["Trigger: git push / pull_request to main"]
    
    subgraph Job1["Job 1: Lint, Test & Validate (ubuntu-latest)"]
        J1_1["actions/checkout@v4"]
        J1_2["Setup Python 3.11 with pip caching"]
        J1_3["Install dependencies: pip install -r requirements.txt"]
        J1_4["Ruff Linter: ruff check src/ api/ tests/"]
        J1_5["Data Pipeline Check: generate, validate, preprocess"]
        J1_6["Model Training Baseline Check: python src/training/train.py"]
        J1_7["Pytest Test Suite: pytest -v"]

        J1_1 --> J1_2 --> J1_3 --> J1_4 --> J1_5 --> J1_6 --> J1_7
    end

    subgraph Job2["Job 2: Build & Verify Docker Image (ubuntu-latest)"]
        J2_1["actions/checkout@v4"]
        J2_2["docker/setup-buildx-action@v3"]
        J2_3["docker/build-push-action@v5 (push: false)"]

        J2_1 --> J2_2 --> J2_3
    end

    Push --> Job1
    Job1 -->|Success (needs)| Job2
```

---

## 3. Detailed Step Breakdown

### Step 1: Linting with Ruff
- **Command:** `ruff check src/ api/ tests/`
- **Role:** Enforces PEP 8 compliance, catches unused imports, detects syntax errors, and validates import ordering.

### Step 2: Data Pipeline Validation
- **Command:**
  ```bash
  python src/data/generate_data.py --config params.yaml
  python src/data/validate_data.py --config params.yaml
  python src/data/preprocess.py --config params.yaml
  ```
- **Role:** Proves that the synthetic data generator, schema validation rules, and feature transformers (`ColumnTransformer`) execute cleanly without runtime crashes.

### Step 3: Model Training Baseline
- **Command:** `python src/training/train.py --config params.yaml`
- **Role:** Confirms that Linear Regression and Random Forest models fit properly, metric calculations succeed, and the fallback pipeline artifact is saved.

### Step 4: Pytest Test Suite
- **Command:** `pytest -v --durations=10`
- **Role:** Executes the full 20-test automated suite covering data validation, preprocessing, model evaluation metrics, API health, prediction, batch scoring, invalid 422 error tracking, and drift detection.

### Step 5: Docker Container Compilation
- **Role:** Runs Docker Buildx to build the production FastAPI image (`Dockerfile`). If any dependency conflict or file path mismatch exists, the CI pipeline fails before broken images reach production.

---

## 4. How to Run the Complete CI Pipeline Locally

You don't need to push to GitHub to verify your code. You can execute the exact equivalent commands locally via the `Makefile`:

```bash
# 1. Format and lint
make lint

# 2. Run data pipeline & training
make pipeline

# 3. Run Pytest suite
make test

# 4. Build Docker container
make docker-build
```

