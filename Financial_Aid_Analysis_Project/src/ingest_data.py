from pathlib import Path

from src.data_ingestion import DataIngestor


def save_dataset(
    root_dir: Path,
    prefixes: tuple[str, ...],
    output_path: Path,
) -> None:
    """Write all matching CSV chunks to one combined CSV file."""
    ingestor = DataIngestor(root_dir=root_dir, prefixes=prefixes)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output_path.open("w", newline="") as f:
            for chunk_index, chunk in enumerate(ingestor.iter_chunks()):
                chunk.to_csv(f, header=chunk_index == 0, index=False)
    except BaseException:
        output_path.unlink(
            missing_ok=True
        )  # Prevents partial files on the even of failure
        raise


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    raw_data_dir = project_dir / "data" / "raw_data"
    combined_dir = project_dir / "data" / "combined"

    print("------Saving Combined Financial_Aid Dataset----\n")
    save_dataset(
        raw_data_dir,
        ("FINAID_",),
        combined_dir / "financial_aid.csv",
    )
    print("------Saving Combined Student Cohort Dataset----\n")
    save_dataset(
        raw_data_dir,
        ("STUDENT_TERM_",),
        combined_dir / "student_cohort.csv",
    )


if __name__ == "__main__":
    main()
