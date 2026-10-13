# Financial Aid Impact Analysis

This project combines financial-aid and student-term records, creates EDA
datasets and an ML master table, and builds reproducible train/test modeling
views. Run the commands below from the `Financial_Aid_Analysis_Project`
directory (the directory containing this README).

## Run the full pipeline

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

### 2. Provide the source data

Place the CSV files under `data/raw_data/` (subdirectories are supported):

| Dataset | Required file | Files combined by prefix |
| --- | --- | --- |
| Static student data | `Student_Static_Data.csv` | Read directly |
| Financial aid | One or more CSVs starting with `FINAID_` | Yes |
| Student term | One or more CSVs starting with `STUDENT_TERM_` | Yes |

The source files must include the columns required by `configs/`. Keep schemas
consistent among files for the same dataset. The `data/` directory is ignored
by Git; each person running the pipeline must have access to the source data.

If combined aid and term files already exist at
`data/combined/financial_aid.parquet` (or `.csv`) and
`data/combined/student_term.parquet` (or `.csv`), the EDA and master-table
steps can read those instead of combining the prefixed CSVs. The static CSV is
still read from `data/raw_data/Student_Static_Data.csv`. To run the explicit
ingestion step below, the prefixed raw CSVs must be present.

### 3. Combine the raw financial-aid and student-term files

```bash
python -m src.preprocessing.ingest_data
```

This recursively discovers `FINAID_*.csv` and `STUDENT_TERM_*.csv`, combines
each set, and writes:

```text
data/combined/financial_aid.parquet
data/combined/student_term.parquet
```

### 4. Create the EDA datasets

```bash
python -m src.preprocessing.eda_dataset_creation
```

This joins the source tables and creates student-type A/B datasets:

```text
data/derived/eda/eda_term_A.parquet
data/derived/eda/eda_term_B.parquet
data/derived/eda/eda_aid_A.parquet
data/derived/eda/eda_aid_B.parquet
```

### 5. Create the ML master table

```bash
python -m src.preprocessing.master_table_creation
```

The output is:

```text
data/derived/master_table.parquet
```

The expected row count is configured in `configs/ml.yaml`. Set
`expected_rows: null` there to disable that source-data check when working with
a different dataset.

### 6. Create modeling train/test views

```bash
python -m src.preprocessing.modeling_dataset
```

This reads the master table, creates a reproducible student-level split, builds
and preprocesses the configured views, and writes:

```text
data/derived/split_v1.parquet
data/derived/views/view_<name>_train.parquet
data/derived/views/view_<name>_test.parquet
data/derived/views/views_config.json
```

Each view file contains its features and a `y` outcome column. The configured
view names, feature columns, and outcomes are in `configs/model_views.yaml`;
the generated `views_config.json` contains the corresponding view settings.

For example, use `view_ftic_ret1_train.parquet` to train or cross-validate on
the FTIC first-year-retention training set. Separate the `y` column from the
features before fitting a model. Use the preprocessor created from the matching
view settings so nullable Parquet values are normalized before imputation.

### 7. Verify the outputs

Each Parquet save is read back and checked for matching shape and dtypes.
Ingestion also checks that the combined row counts match the source CSV row
counts. A failed write verification removes the incomplete output and raises
an error.

## Run the tests

Install the development dependencies and run the test suite:

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
```
