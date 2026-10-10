from pathlib import Path

import pandas as pd


def save(df: pd.DataFrame, path: str | Path) -> None:
    """Save a dataframe as Parquet and verify its shape and dtypes round-trip."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(output_path, index=False)
        restored = pd.read_parquet(output_path)
        assert restored.shape == df.shape, (
            f"Shape mismatch after saving {output_path}: "
            f"expected {df.shape}, got {restored.shape}"
        )
        assert restored.dtypes.equals(df.dtypes), (
            f"Dtype mismatch after saving {output_path}: "
            f"expected {df.dtypes.to_dict()}, got {restored.dtypes.to_dict()}"
        )
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
