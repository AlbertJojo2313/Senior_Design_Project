from pathlib import Path

import pandas as pd
from src.preprocessing.data_ingestion import DataIngestor
from src.utils.save import save


def save_dataset(
    root_dir: Path,
    prefixes: tuple[str, ...],
    output_path: Path,
) -> None:
    """Write all matching CSV chunks to one combined Parquet file."""
    ingestor = DataIngestor(root_dir=root_dir, prefixes=prefixes)
    source_row_counts = ingestor.count_rows_by_file()
    expected_rows = sum(source_row_counts.values())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        combined = ingestor.load()
        save(combined, output_path)
        combined_rows = len(pd.read_parquet(output_path))
        source_counts = ", ".join(
            f"{path.name}: {count}" for path, count in source_row_counts.items()
        )
        assert combined_rows == expected_rows, (
            f"Row count mismatch for {output_path}: source files contain "
            f"{expected_rows} rows ({source_counts}), but the combined dataset "
            f"contains {combined_rows} rows."
        )
    except BaseException:
        output_path.unlink(
            missing_ok=True
        )  # Prevents partial files on the even of failure
        raise


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    raw_data_dir = project_dir / "data" / "raw_data"
    combined_dir = project_dir / "data" / "combined"

    print("------Saving Combined Financial_Aid Dataset----\n")
    save_dataset(
        raw_data_dir,
        ("FINAID_",),
        combined_dir / "financial_aid.parquet",
    )
    print("------Saving Combined Student Cohort Dataset----\n")
    save_dataset(
        raw_data_dir,
        ("STUDENT_TERM_",),
        combined_dir / "student_term.parquet",
    )


if __name__ == "__main__":
    main()
