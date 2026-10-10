# Data Ingestion

## Purpose

`DataIngestor` discovers and reads the yearly financial-aid and student-term
CSV files. The ingestion command combines each source family into a Parquet
file while checking that the number of output rows equals the sum of the
matching source-file row counts.

## Inputs and outputs

Place source CSVs in `data/raw_data/` or in subdirectories below it. Discovery
is recursive and matches filenames by prefix:

| Source files | Prefix | Combined output |
| --- | --- | --- |
| Financial-aid extracts | `FINAID_` | `data/combined/financial_aid.parquet` |
| Student-term extracts | `STUDENT_TERM_` | `data/combined/student_term.parquet` |

The static student table is read separately by dataset creation from
`data/raw_data/Student_Static_Data.csv`.

## How ingestion works

1. `DataIngestor.discover_files()` finds matching CSVs in sorted path order and
   reports a clear error when the input directory or matching files are absent.
2. CSV row counts are measured per source file without loading each entire file.
3. PyArrow reads each CSV in batches. Columns named `STUDENT_ID` and
   `ITEM_TYPE`, when present, are explicitly read as text to preserve identifier
   and item-code values.
4. `src/preprocessing/ingest_data.py` combines the batches into a pandas
   DataFrame and saves it through `src/utils/save.py`.
5. The output is reloaded; the row count is compared with the source total, and
   the shared save helper checks shape and dtypes. If writing or validation
   fails, the incomplete output is removed and the error is raised.

The data-ingestion reader yields batches as its workers finish; row ordering
across source files is not a guaranteed part of the output contract. Consumers
should rely on keys and sort explicitly when chronological or stable ordering
is required.

## Run ingestion

From the project root, run:

```bash
python -m src.preprocessing.ingest_data
```

This command writes or replaces both combined Parquet files. To build EDA and
ML datasets after ingestion, see [Dataset Creation](dataset_creation.md).

## Related implementation and tests

- Reader: `src/preprocessing/data_ingestion.py`
- Command: `src/preprocessing/ingest_data.py`
- Parquet save and verification: `src/utils/save.py`
- Synthetic tests: `tests/test_data_ingestion.py`

Run the ingestion tests with:

```bash
python -m unittest discover -s tests -p "test_data_ingestion.py" -v
```
