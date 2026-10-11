from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing.dataset_creation import (
    load_config,
    load_sources,
    resolve_project_dir,
)
from src.utils.save import save


class EDADatasetCreation:
    def __init__(
        self,
        config_path: str | Path | None = None,
        project_dir: str | Path | None = None,
    ) -> None:
        self.project_dir = resolve_project_dir(project_dir)
        self.cfg: dict[str, Any] = load_config(
            "eda.yaml", self.project_dir, config_path
        )
        self.logger = logging.getLogger(__name__)

    def log(self, message: str) -> None:
        self.logger.info(message)

    def load_sources(self) -> dict[str, pd.DataFrame]:
        return load_sources(self.project_dir, self.logger)

    def create_datasets(
        self, sources: dict[str, pd.DataFrame] | None = None
    ) -> dict[str, pd.DataFrame]:
        data = sources if sources is not None else self.load_sources()
        static = data["student_static"]
        term = data["student_term"]
        aid = data["financial_aid"]

        eda_term = static.merge(
            term,
            on="STUDENT_ID",
            how="left",
            validate="one_to_many",
            indicator=True,
        )
        assert eda_term["_merge"].eq("both").all(), (
            "student_static rows without a student_term match"
        )
        eda_term = eda_term.drop(columns="_merge")

        eda_aid = static.merge(
            aid,
            on="STUDENT_ID",
            how="left",
            validate="one_to_many",
            indicator=True,
        )
        assert eda_aid["_merge"].eq("both").all(), (
            "student_static rows without a financial_aid match"
        )
        eda_aid = eda_aid.drop(columns="_merge")

        outputs: dict[str, pd.DataFrame] = {}
        for dataset_name, frame in (("eda_term", eda_term), ("eda_aid", eda_aid)):
            for student_type in ("A", "B"):
                split = frame[frame["STUDENT_TYPE"].eq(student_type)].copy()
                assert split["STUDENT_TYPE"].nunique() <= 1
                outputs[f"{dataset_name}_{student_type}"] = split
            assert len(outputs[f"{dataset_name}_A"]) + len(
                outputs[f"{dataset_name}_B"]
            ) == len(frame), f"{dataset_name} contains an unexpected STUDENT_TYPE"

        output_dir = self.project_dir / self.cfg["output_dir"]
        for name, frame in outputs.items():
            save(frame, output_dir / f"{name}.parquet")
        return outputs

    def output_path(self, name: str) -> Path:
        return self.project_dir / self.cfg["output_dir"] / f"{name}.parquet"


def main() -> None:
    EDADatasetCreation().create_datasets()


if __name__ == "__main__":
    main()
