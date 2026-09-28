# Financial Aid Impact Analysis

This project combines yearly financial-aid and student-cohort CSV files into
two datasets for analysis. The ingestion code is in `src/ingest_data.py`; it
discovers matching CSV files recursively, reads them in chunks, and writes the
combined files under `data/combined/`.

## Run the ingestion from start to finish

Run these commands from the `Financial_Aid_Analysis_Project` directory (the
directory containing this README).

### 1. Set up Python

Python 3.12 or later is required. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project and its dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

### 2. Add the source CSV files

Create the input directory:

```bash
mkdir -p data/raw_data
```

Copy the source CSV files into `data/raw_data/` (or its subdirectories). The
ingestion searches recursively, selecting files by filename prefix:

| Dataset | Required filename prefix | Combined output |
| --- | --- | --- |
| Financial aid | `FINAID_` | `data/combined/financial_aid.csv` |
| Student cohort | `STUDENT_TERM_` | `data/combined/student_cohort.csv` |

For example, `FINAID_2025.csv` and `STUDENT_TERM_2025.csv` both match. Each
dataset needs at least one matching CSV file. Keep the columns and their types
consistent among files within each dataset. The `data/` directory is ignored
by Git, so input data and generated outputs are not committed.

### 3. Run ingestion

From the project directory, run:

```bash
python -m src.ingest_data
```

The script creates `data/combined/` if needed and writes both combined CSVs.
Running it again replaces those output files with newly generated datasets.

### 4. Confirm the result

Check that these files were created:

```text
data/combined/financial_aid.csv
data/combined/student_cohort.csv
```

For each output, ingestion compares its data-row count with the total row count
of the matching source files. A mismatch raises an error and removes the
incomplete output file. Missing input directories or missing matching CSVs
also raise an error.

## Run the tests

To run the ingestion tests, install the development dependencies and use
Python's unittest runner:

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
```