# Master Table Creation

## Purpose and scope

`MasterTableCreation` builds a student-level master table with one row per
financial-aid recipient. It combines student-static attributes with aid
indicators, selected first-term information, and term-year summaries.

The master table avoids giving students with more source records extra rows in
later analyses. It is an input to downstream analysis and modeling-view
creation; this script does not create train/test splits or fit models.

## Inputs and prerequisites

The pipeline reads:

- `data/raw_data/Student_Static_Data.csv`
- Combined financial-aid data from `data/combined/financial_aid.parquet` or
  `data/combined/financial_aid.csv`, when present
- Combined student-term data from `data/combined/student_term.parquet` or
  `data/combined/student_term.csv`, when present

For either combined dataset, Parquet is preferred over CSV. If neither combined
file exists, the pipeline reads matching raw CSVs recursively from
`data/raw_data/`: filenames beginning with `FINAID_` for aid and
`STUDENT_TERM_` for term data. The static file is read directly from
`data/raw_data/Student_Static_Data.csv`.

`STUDENT_ID` is required in all three source tables. IDs are converted to
trimmed text, and the static table must have one row per ID. Student-term and
static records are restricted to IDs present in financial aid; the resulting
master table therefore represents aid recipients, not every student in the
static file.

The transformations also rely on aid fields such as `AID_YEAR`,
`FA_SOURCE_DESCR`, and `COHORT_YEAR`; term fields such as `CF_BOE_TERM_ID`,
`ACAD_YEAR`, `Credits_Transfered`, `CUM_TOTHRS_EARN_BEGIN_TERM`,
`Credits_Earned_In_Term`, and `CF_BOE_TERM_CRS_LD`; and the columns selected
by `first_cols` and `output_columns` in `configs/ml.yaml`.

## Build process

Run the script from the project directory:

```bash
python -m src.preprocessing.master_table_creation
```

The script performs these steps:

1. **Prepare aid indicators.** Static cohort year is used to calculate each
   aid record's year relative to the student's cohort. Aid records outside the
   configured `aid_years` range are excluded; a record before the configured
   first year is treated as an error. Aid source descriptions are normalized
   and converted into binary columns named
   `aid_yr{year}_src_{source}`. The configured `aid_sources` list controls the
   expected output columns; an unexpected source is reported as an error.
2. **Clean and map term records.** Duplicate rows for the same student and
   `CF_BOE_TERM_ID` are reduced to the most complete row. The pipeline checks
   that academic years do not move backward within a student's terms.
   `term_yr` maps a term to the student's year relative to the static cohort:
   academic-year end year minus cohort end year plus one.
3. **Collect first-term attributes.** The earliest term, ordered by student
   and term ID, supplies the columns configured by `first_cols`. Maximum
   transfer credits are also calculated for each student.
4. **Summarize terms by year.** Terms within the configured `term_years` range
   are aggregated per student and year into:
   - `n_terms`: distinct term count
   - `credits_enrolled`: sum of enrolled credits
   - `credits_earned`: sum of earned credits
   - `end_cum_gpa`: last cumulative GPA in the year
   - `end_ucf_gpa`: last UCF GPA in the year
   - `end_cum_hrs`: last cumulative-hours value, calculated from beginning
     cumulative hours plus credits earned in the term
5. **Merge and clean the master table.** Student-static records are joined
   with the aid indicators, first-term attributes, and yearly summaries using
   one-to-one merge validation. The pipeline checks that the student count
   stays stable, that IDs remain unique, and that each aid recipient has a
   first-term record. It adds cohort-mismatch flags, fills absent yearly term
   counts with zero, applies configured null and type conversions, and selects
   columns in the order specified by `output_columns`.

The term summaries describe observed terms by cohort-relative year. Whether a
summary is a valid feature for a particular prediction depends on the
prediction outcome and time horizon; downstream modeling views should only
include information available at the prediction point.

## Configuration

Settings are in `configs/ml.yaml`:

| Setting | Purpose |
| --- | --- |
| `aid_years` | Inclusive cohort-relative aid-year range to include |
| `term_years` | Inclusive cohort-relative term-year range to summarize |
| `aid_sources` | Aid sources expected when creating fixed indicator columns |
| `term_drop_cols` | Term columns removed before term processing |
| `first_cols` | Term columns retained from each student's earliest term |
| `grad_flag_cols` | Graduation-count columns treated as nullable integer flags |
| `sat_zero_is_missing` | Whether zero-valued SAT scores are recoded to missing |
| `post_merge_drop_cols` | Temporary or unwanted merged columns to remove |
| `output_columns` | Ordered allow-list of columns in the final table |
| `output_path` | Project-relative path for the master-table Parquet file |
| `expected_rows` | Optional expected student count; set to `null` to disable this check |

## Output and verification

The current configuration writes the master table to
`data/derived/master_table.parquet`. The configured `output_columns` list
determines its schema and column order.

The output is written through `src/utils/save.py`, which reloads the Parquet
file and checks that its shape and column data types match the in-memory
DataFrame. If writing or verification fails, the incomplete output is removed
and the error is raised.

## Troubleshooting

- **Missing source file:** Confirm the combined files or matching raw CSVs
  exist at the expected paths; confirm the static file is named
  `Student_Static_Data.csv`.
- **Missing column:** Check source schemas and the configured `first_cols`,
  `term_drop_cols`, `grad_flag_cols`, and `output_columns`.
- **Unexpected aid source:** Add the normalized source name to `aid_sources`
  only if it belongs in the output; otherwise correct the source data.
- **Row-count mismatch:** Check `expected_rows` against the current aid-recipient
  population, or set it to `null` if no fixed population-size check is needed.
- **Duplicate or invalid term records:** Check for duplicate student/term keys,
  missing join IDs, or academic years that go backward for a student.
