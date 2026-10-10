# Financial Aid Impact Analysis

This project combines yearly financial-aid and student-term data, then builds
EDA and machine-learning datasets for analysis. Ingestion and dataset creation
are implemented in `src/preprocessing/`; generated Parquet files are written
under `data/combined/` and `data/derived/`.

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
| Financial aid | `FINAID_` | `data/combined/financial_aid.parquet` |
| Student term | `STUDENT_TERM_` | `data/combined/student_term.parquet` |

For example, `FINAID_2025.csv` and `STUDENT_TERM_2025.csv` both match. Each
dataset needs at least one matching CSV file. Keep the columns and their types
consistent among files within each dataset. The `data/` directory is ignored
by Git, so input data and generated outputs are not committed.

### 3. Run ingestion

From the project directory, run:

```bash
python -m src.preprocessing.ingest_data
```

The script creates `data/combined/` if needed and writes both combined
Parquet files. Running it again replaces those output files.

### 4. Create analysis datasets

With the combined datasets and `Student_Static_Data.csv` in place, create the
four EDA splits and the ML master dataset:

```bash
python -m src.preprocessing.eda_dataset_creation
python -m src.preprocessing.ml_dataset_creation
```

The EDA outputs are written to `data/derived/eda/`; the ML output is
`data/derived/ml_dataset.parquet`. Configuration is in `configs/eda.yaml` and
`configs/ml.yaml`. The ML configuration's `expected_rows` is a source-data
check; set it to `null` to disable that check.

### 5. Confirm the result

Check that these files were created:

```text
data/combined/financial_aid.parquet
data/combined/student_term.parquet
data/derived/eda/eda_term_A.parquet
data/derived/eda/eda_term_B.parquet
data/derived/eda/eda_aid_A.parquet
data/derived/eda/eda_aid_B.parquet
data/derived/ml_dataset.parquet
```

Ingestion compares combined row counts with the total from its matching source
files. All Parquet outputs are reloaded and checked for matching shape and
dtypes. A failed verification removes the incomplete output. Missing inputs
also raise an error.

## Run the tests

Install the development dependencies and run the synthetic fixture tests:

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
```