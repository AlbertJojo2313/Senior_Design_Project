# Dataset Creation

<!-- dataset-creation-feature:start -->

## Purpose

The dataset-creation pipeline turns financial-aid, student-term, and student
static source tables into two kinds of analysis-ready outputs:

- **EDA datasets:** aid and term records joined separately to student static
  attributes, each split into transfer (`STUDENT_TYPE = A`) and FTIC
  (`STUDENT_TYPE = B`) populations.
- **ML dataset:** one row per aid student, with aid indicators and selected
  first-term and year-level summaries.

The pipeline prepares data for analysis; it does not split train/test data,
impute missing values, encode features, or scale data. Those operations belong
to later modeling or view-building steps.

## Data flow

```mermaid
flowchart TD
    subgraph Sources
        AidCSV["Yearly financial-aid CSVs"]
        TermCSV["Yearly student-term CSVs"]
        StaticCSV["Student_Static_Data.csv"]
    end

    AidCSV --> Ingest["DataIngestor<br/>PyArrow CSV reader"]
    TermCSV --> Ingest
    Ingest --> CombinedAid["data/combined/financial_aid.parquet"]
    Ingest --> CombinedTerm["data/combined/student_term.parquet"]

    CombinedAid --> Shared["DatasetCreation<br/>load and normalize sources"]
    CombinedTerm --> Shared
    StaticCSV --> Shared
    Shared --> students["Keep aid students<br/>normalize STUDENT_ID"]

    students --> EDA["EDADatasetCreation"]
    EDA --> EdaTerm["Static + term<br/>one-to-many"]
    EDA --> EdaAid["Static + aid<br/>one-to-many"]
    EdaTerm --> EdaSplit["Split by STUDENT_TYPE"]
    EdaAid --> EdaSplit
    EdaSplit --> EdaFiles["Four EDA Parquet files<br/>A/transfer and B/FTIC"]

    students --> ML["MLDatasetCreation"]
    ML --> AidWide["Aid long-to-wide<br/>year/source indicators"]
    ML --> TermClean["Deduplicate and date terms"]
    TermClean --> FirstTerm["First-term attributes"]
    TermClean --> YearSummary["Per-year summaries"]
    AidWide --> Master["One-to-one student-level merges"]
    FirstTerm --> Master
    YearSummary --> Master
    students --> Master
    Master --> Cleanup["Cleanup, flags, and nullable types"]
    Cleanup --> MLFile["data/derived/ml_dataset.parquet"]
```

Every Parquet output is written through `src/utils/save.py`. The save helper
reloads the file and checks its shape and dtypes against the in-memory
DataFrame; a failed write or check removes the incomplete output.

## Architecture and design rationale

### Shared source preparation

`DatasetCreation` centralizes configuration loading, source loading, key
normalization, and student filtering. Both EDA and ML therefore start from
the same normalized student population instead of implementing slightly
different versions of these rules.

Student identifiers and aid item codes are read as text so leading zeros and
nulls are not lost to numeric inference. `STUDENT_ID` values are stripped
before matching. The static table is required to contain one row per student,
and non-students are filtered from static and term data with a log message
recording the number removed.

### Keep long tables separate for EDA

Financial aid and student terms are both one-to-many tables by student. Joining
them directly would multiply a student's aid records by their term records and
inflate counts and totals. The EDA builder instead joins each table to static
attributes independently, validates the one-to-many relationship, checks that
the expected matches exist, and then creates the A/B splits(Transfer/FTIC).

### Build the ML table from student-level components

The ML output has one row per student, while raw aid and term data can have
many rows per student. The builder first reduces each source to a student-level
component—aid indicators, first-term attributes, and year summaries—then joins
those components to the one-row-per-student static table with one-to-one
validation. This keeps row counts predictable and makes duplicate-key or
unexpected merge behavior fail visibly.

Static cohort year is the source of truth for aid-year and term-year
calculations. Source cohort disagreements are retained as mismatch flags
rather than silently overwriting the source values. Missing first-term fields
stay missing instead of being filled from later terms. A missing year-level
term count becomes zero after the merge, while missing outcome measures remain
missing.

### Configuration and storage

EDA and ML each have a YAML config (`configs/eda.yaml` and `configs/ml.yaml`).
This keeps output paths, year ranges, selected columns, and cleanup rules
explicit and adjustable without duplicating the pipeline implementation.
`expected_rows` provides an optional check on the ML student population; set
it to `null` to skip that source-specific expectation.
The ML output is restricted to the ordered allow-list in `output_columns`, so
intermediate or unused fields do not leak into `ml_dataset.parquet`. Graduation
count column names are preserved from the source table.

Aid source groups are preserved independently. In particular, the ML config
includes `other` in `aid_sources`, so it creates its own `aid_yrN_src_other`
indicator instead of combining those records with private aid.

Parquet is used for combined and derived datasets because it preserves column
types, nullable integer flags, text codes, and null values more reliably than
CSV. Pandas is used for the dataset transformations—filtering, joins,
reshaping, summaries, and type conversion—while PyArrow reads CSV ingestion
inputs in chunks.

## Main components

| Component | Responsibility |
| --- | --- |
| `src/preprocessing/data_ingestion.py` | Recursively discovers matching CSVs and reads them in chunks; preserves `STUDENT_ID` and `ITEM_TYPE` as text. |
| `src/preprocessing/ingest_data.py` | Combines raw aid and term CSVs into Parquet outputs and checks row counts. |
| `src/preprocessing/dataset_creation.py` | Loads configured inputs, normalizes keys, and restricts static and term tables to aid students. |
| `src/preprocessing/eda_dataset_creation.py` | Creates separate aid/static and term/static joins and their A/B splits. |
| `src/preprocessing/ml_dataset_creation.py` | Creates the one-row-per-student ML master table. |
| `src/utils/save.py` | Writes and round-trip verifies Parquet outputs. |

## Run the pipeline

Run these commands from the project directory:

```bash
python -m src.preprocessing.ingest_data
python -m src.preprocessing.eda_dataset_creation
python -m src.preprocessing.ml_dataset_creation
```

The dataset-creation commands can also load existing combined Parquet files or
CSV files. When combined files are absent, they load matching raw aid and term
CSVs directly. The static table is loaded from
`data/raw_data/Student_Static_Data.csv`.

## Validation

The tests use small synthetic DataFrames and temporary CSV/Parquet files; they
do not require access to the project's student records. They cover key
preservation and student filtering, aid indicator columns, duplicate-term
selection, first-term null behavior, term-year mapping, year summaries,
master-table row invariants, cleanup types, EDA splits, and Parquet ingestion.

Run them with:

```bash
python -m unittest discover -s tests -v
```

<!-- dataset-creation-feature:end -->
