# Get all the datasets and aggregates them into the one section (Final Goal)
# Current version: aggregate the given datasets into 3 separate chunks
#  - financial_aid, student cohorts, and static that is just reading one file


from concurrent.futures import ThreadPoolExecutor

import pandas as pd


def retrieve_files(file_paths: list[str]) -> pd.DataFrame:

    with ThreadPoolExecutor(max_workers=4) as pool:
        dfs = list(pool.map(pd.read_excel, file_paths))

    df = pd.concat(dfs, ignore_index=False)

    return df


def main() -> None:
    financial_aid_file_paths = [
        "../data/raw_data/FINAID_2021.csv",
        "../data/raw_data/FINAID_2022.csv",
        "../data/raw_data/FINAID_2023",
        "../data/raw_data/FINAID_2024.csv",
        "../data/raw_data/FINAID_2025.csv",
        "../data/raw_data/FINAID_2026.csv",
    ]
    fin_df = retrieve_files(financial_aid_file_paths)
