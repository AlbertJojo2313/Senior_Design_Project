# Data Ingestion Script
# Loads different sources of data => Preprocessing script
# Future Plan to make a data ingestion class

from concurrent.futures import ThreadPoolExecutor

import pandas as pd


def retrieve_files(file_paths: list[str]) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=4) as pool:
        dfs = list(pool.map(pd.read_csv, file_paths))
    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    financial_aid_file_paths = [
        "../data/raw_data/FINAID_2021.csv",
        "../data/raw_data/FINAID_2022.csv",
        "../data/raw_data/FINAID_2023.csv",  # was missing .csv
        "../data/raw_data/FINAID_2024.csv",
        "../data/raw_data/FINAID_2025.csv",
        "../data/raw_data/FINAID_2026.csv",
    ]
    student_cohort_file_paths = [
        "../data/raw_data/STUDENT_TERM_2019.csv",
        "../data/raw_data/STUDENT_TERM_2020.csv",
        "../data/raw_data/STUDENT_TERM_2021.csv",
        "../data/raw_data/STUDENT_TERM_2022.csv",
        "../data/raw_data/STUDENT_TERM_2023.csv",
        "../data/raw_data/STUDENT_TERM_2024.csv",
        "../data/raw_data/STUDENT_TERM_2025.csv",
    ]

    print("---------- Data Ingestion Started-------\n")
    print("Getting Data from FINANCIAL DATASETS\n")
    fin_df = retrieve_files(financial_aid_file_paths)
    print("Getting Data from STUDENT COHORT DATASETS")

    student_cohort_df = retrieve_files(student_cohort_file_paths)

    student_cohort_df.to_csv("../data/combined/student_cohort.csv", index=False)
    fin_df.to_csv("../data/combined/financial_aid.csv", index=False)


if __name__ == "__main__":
    main()
