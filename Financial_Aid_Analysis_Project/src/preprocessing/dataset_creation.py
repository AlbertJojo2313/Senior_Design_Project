"""Shared configuration and input-loading functions for dataset scripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from src.preprocessing.data_ingestion import DataIngestor


def resolve_project_dir(project_dir: str | Path | None = None) -> Path:
    if project_dir is not None:
        return Path(project_dir).resolve()
    return Path(__file__).resolve().parents[2]


def load_config(
    config_name: str,
    project_dir: Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    path = (
        Path(config_path)
        if config_path is not None
        else project_dir / "configs" / config_name
    )
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return config


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"STUDENT_ID": str, "ITEM_TYPE": str})


def load_combined(
    project_dir: Path,
    dataset_name: str,
    source_prefix: tuple[str, ...],
) -> pd.DataFrame:
    combined_dir = project_dir / "data" / "combined"
    for suffix in (".parquet", ".csv"):
        path = combined_dir / f"{dataset_name}{suffix}"
        if path.is_file():
            return read_table(path)
    raw_dir = project_dir / "data" / "raw_data"
    return DataIngestor(root_dir=raw_dir, prefixes=source_prefix).load()


def load_sources(project_dir: Path, logger: logging.Logger) -> dict[str, pd.DataFrame]:
    """Load sources, normalize student IDs, and restrict to aid recipients."""
    aid = load_combined(project_dir, "financial_aid", ("FINAID_",))
    term = load_combined(project_dir, "student_term", ("STUDENT_TERM_",))
    static = read_table(project_dir / "data" / "raw_data" / "Student_Static_Data.csv")

    if "Aid_Year" in aid and "AID_YEAR" not in aid:
        aid = aid.rename(columns={"Aid_Year": "AID_YEAR"})

    tables = {
        "financial_aid": aid,
        "student_term": term,
        "student_static": static,
    }
    for table_name, frame in tables.items():
        if "STUDENT_ID" not in frame:
            raise KeyError(f"{table_name} is missing required STUDENT_ID column")
        frame["STUDENT_ID"] = frame["STUDENT_ID"].astype(str).str.strip()
    if not static["STUDENT_ID"].is_unique:
        raise ValueError("student_static must have one row per STUDENT_ID")

    recipients = set(aid["STUDENT_ID"].dropna())
    for table_name in ("student_term", "student_static"):
        frame = tables[table_name]
        original_ids = set(frame["STUDENT_ID"].dropna())
        tables[table_name] = frame[frame["STUDENT_ID"].isin(recipients)].copy()
        logger.info(
            "%s: removed %s non-recipient student IDs",
            table_name,
            len(original_ids - recipients),
        )
    return tables
