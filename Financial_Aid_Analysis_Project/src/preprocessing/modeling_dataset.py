from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from src.preprocessing.dataset_creation import (
    load_config,
    resolve_project_dir,
)
from src.preprocessing.master_table_creation import MasterTableCreation
from src.utils.save import save


def _to_float_array(values: pd.DataFrame | np.ndarray) -> np.ndarray:
    return pd.DataFrame(values).to_numpy(dtype=np.float64, na_value=np.nan)


def _to_object_array(values: pd.DataFrame | np.ndarray) -> np.ndarray:
    return pd.DataFrame(values).to_numpy(dtype=object, na_value=np.nan)


class ModelingDatasetCreation:
    """Create reproducible train/test views from the ML master dataset."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        project_dir: str | Path | None = None,
    ) -> None:
        self.project_dir = resolve_project_dir(project_dir)
        self.cfg: dict[str, Any] = load_config(
            "model_views.yaml", self.project_dir, config_path
        )
        self.logger = logging.getLogger(__name__)

    def log(self, message: str) -> None:
        self.logger.info(message)

    def create_datasets(
        self, sources: dict[str, pd.DataFrame] | None = None
    ) -> dict[str, dict[str, Any]]:
        master = self.prepare_master(
            sources["master"] if sources is not None else self.load_master()
        )
        split = self.build_split(master)
        save(split, self.project_dir / self.cfg["split_path"])

        datasets: dict[str, dict[str, Any]] = {}
        views_config = self.view_configs()
        output_dir = self.project_dir / self.cfg["views_output_dir"]
        for name, cfg in views_config.items():
            dataset = self.build_view(master, split, cfg)
            datasets[name] = dataset
            for part in ("train", "test"):
                features = dataset[f"X_{part}"].copy()
                features[cfg["cat"]] = features[cfg["cat"]].astype("string")
                save(
                    features.assign(y=dataset[f"y_{part}"]),
                    output_dir / f"view_{name}_{part}.parquet",
                )
            self.log(
                f"{name:11} train {len(dataset['y_train']):6}  "
                f"test {len(dataset['y_test']):5}  "
                f"rate {dataset['y_train'].mean():.3f} / "
                f"{dataset['y_test'].mean():.3f}  "
                f"features {dataset['X_train_transformed'].shape[1]}"
            )

        config_path = self.project_dir / self.cfg["views_config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(views_config, config_file, indent=2)
        self.log(f"Saved {len(datasets)} modeling views")
        return datasets

    def load_master(self) -> pd.DataFrame:
        """Load the configured master parquet, creating it if it is absent."""
        input_path = self.project_dir / self.cfg["input_path"]
        if not input_path.is_file():
            MasterTableCreation(project_dir=self.project_dir).create_datasets()
        if not input_path.is_file():
            raise FileNotFoundError(
                f"ML master dataset was not created at {input_path}"
            )
        return pd.read_parquet(input_path)

    def build_split(self, master: pd.DataFrame) -> pd.DataFrame:
        student_ids = master["STUDENT_ID"]
        if student_ids.isna().any():
            raise ValueError("STUDENT_ID must not contain null values")
        if student_ids.duplicated().any():
            raise ValueError("STUDENT_ID must be unique before creating the split")

        strata = (
            master["STUDENT_TYPE"].astype(str)
            + "_"
            + master["RETAINED_YR1"].astype(str)
        )
        train_ids, test_ids = train_test_split(
            student_ids,
            test_size=self.cfg["test_size"],
            random_state=self.cfg["random_state"],
            stratify=strata,
        )
        train_id_set = set(train_ids)
        test_id_set = set(test_ids)
        if train_id_set & test_id_set:
            raise AssertionError("Train and test splits contain overlapping IDs")
        if train_id_set | test_id_set != set(student_ids):
            raise AssertionError("Train and test splits do not cover all student IDs")

        split = pd.DataFrame({"STUDENT_ID": student_ids})
        split["split"] = (
            split["STUDENT_ID"].isin(train_id_set).map({True: "train", False: "test"})
        )
        return split

    def build_view(
        self,
        master: pd.DataFrame,
        split: pd.DataFrame,
        cfg: dict[str, Any],
    ) -> dict[str, Any]:
        eligible = master["STUDENT_TYPE"].eq(cfg["type"])
        for column in cfg["require"]:
            eligible &= master[column].eq(1)
        eligible &= master[cfg["treatment"]].sum(axis=1).gt(0)
        eligible &= master[cfg["outcome"]].notna()
        df = master.loc[eligible].merge(
            split, on="STUDENT_ID", how="left", validate="one_to_one"
        )
        if df["split"].isna().any():
            raise ValueError("Every view row must have a train/test split assignment")

        flags = cfg["flags"] + cfg["treatment"]
        X = df[cfg["num"] + cfg["cat"] + flags].copy()
        for column in cfg["num"] + flags:
            X[column] = X[column].to_numpy(dtype=np.float64, na_value=np.nan)
        for column in cfg["cat"]:
            X[column] = pd.Series(
                X[column].to_numpy(dtype=object, na_value=np.nan),
                index=X.index,
                dtype=object,
            )
        y = df[cfg["outcome"]].astype(int)

        train = df["split"].eq("train")
        X_train, X_test = X.loc[train], X.loc[~train]
        y_train, y_test = y.loc[train], y.loc[~train]
        if y_train.empty or y_test.empty:
            raise ValueError(
                f"View for outcome {cfg['outcome']!r} and type {cfg['type']!r} "
                "must have examples in both train and test splits"
            )

        preprocessor = self.make_preprocessor(cfg)
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "preprocessor": preprocessor,
            "X_train_transformed": X_train_transformed,
            "X_test_transformed": X_test_transformed,
        }

    @staticmethod
    def make_preprocessor(cfg: dict[str, Any]) -> ColumnTransformer:
        num_pipe = make_pipeline(
            FunctionTransformer(_to_float_array, validate=False),
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
        )
        cat_pipe = make_pipeline(
            FunctionTransformer(_to_object_array, validate=False),
            SimpleImputer(strategy="constant", fill_value="missing"),
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=30,
                sparse_output=False,
            ),
        )
        return make_column_transformer(
            (num_pipe, cfg["num"]),
            (cat_pipe, cfg["cat"]),
            ("passthrough", cfg["flags"] + cfg["treatment"]),
        )

    def view_configs(self) -> dict[str, dict[str, Any]]:
        entry_load = (
            self.cfg["entry_load_columns"] if self.cfg["include_entry_load"] else []
        )
        return {
            name: {
                **self.cfg["cohorts"][view["cohort"]],
                "flags": entry_load,
                "outcome": view["outcome"],
                "treatment": [
                    f"aid_yr{year}_src_{source}"
                    for year in view["aid_years"]
                    for source in self.cfg["aid_sources"]
                ],
                "require": view["require"],
            }
            for name, view in self.cfg["views"].items()
        }

    def prepare_master(self, master: pd.DataFrame) -> pd.DataFrame:
        """Add derived features and check that the configured views can be built."""
        master = master.copy()
        views = self.view_configs()
        if any("CIP2" in cfg["cat"] for cfg in views.values()):
            if "Entry_CIP" not in master:
                raise KeyError("Columns not in ML master dataset: ['Entry_CIP']")
            master["CIP2"] = master["Entry_CIP"].astype("string").str[:2]

        needed = {
            column
            for cfg in views.values()
            for column in (
                cfg["num"]
                + cfg["cat"]
                + cfg["flags"]
                + cfg["treatment"]
                + [cfg["outcome"]]
                + cfg["require"]
            )
        }
        needed.update(("STUDENT_ID", "STUDENT_TYPE", "RETAINED_YR1"))
        missing = needed - set(master.columns)
        if missing:
            raise KeyError(f"Columns not in ML master dataset: {sorted(missing)}")
        return master

    def aid_columns(self, year: int) -> list[str]:
        return [f"aid_yr{year}_src_{source}" for source in self.cfg["aid_sources"]]

    def history_columns(self, year: int) -> list[str]:
        """Term-history summaries from the years before `year` (pre-treatment)."""
        return [
            f"yr{prior}_{metric}"
            for prior in range(1, year)
            for metric in self.cfg["impact_history_metrics"]
        ]

    def prepare_impact_master(self, master: pd.DataFrame) -> pd.DataFrame:
        """Add CIP2 and cumulative retention outcomes; check impact columns exist."""
        master = master.copy()
        if "Entry_CIP" in master:
            master["CIP2"] = master["Entry_CIP"].astype("string").str[:2]

        # Cumulative outcomes replace filtering on later retention, which would
        # drop the students that aid may have kept enrolled.
        retained = ["RETAINED_YR1", "RETAINED_YR2", "RETAINED_YR3"]
        if set(retained) <= set(master.columns):
            yr1, yr2, yr3 = (master[column].astype("boolean") for column in retained)
            master["RETAINED_THRU_YR2"] = (yr1 & yr2).astype("Int8")
            master["RETAINED_THRU_YR3"] = (yr1 & yr2 & yr3).astype("Int8")

        needed = {"STUDENT_ID", "STUDENT_TYPE"}
        for cohort in self.cfg["impact_cohorts"].values():
            needed.update(cohort["num"] + cohort["cat"])
            for year, outcomes in cohort["outcomes"].items():
                needed.update(outcomes)
                needed.update(self.history_columns(year))
                needed.update(
                    column for prior in range(1, year + 1)
                    for column in self.aid_columns(prior)
                )
                needed.update(f"RETAINED_YR{prior}" for prior in range(1, year))
        missing = needed - set(master.columns)
        if missing:
            raise KeyError(f"Columns not in ML master dataset: {sorted(missing)}")
        return master

    @staticmethod
    def build_impact_view(
        population: pd.DataFrame,
        num: list[str],
        cat: list[str],
        prior_aid: list[str],
        treatment: str,
        concurrent: list[str],
        outcomes: list[str],
    ) -> pd.DataFrame:
        """Full-sample view: treatment flag, raw covariates, missing flags, outcomes."""
        missing_flags = pd.DataFrame(
            {
                f"{column}_missing": population[column].isna().astype("Int8")
                for column in num
            },
            index=population.index,
        )
        view = pd.concat(
            [
                population[["STUDENT_ID"]],
                population[treatment].astype("Int8").rename("treated"),
                population[num + cat + prior_aid],
                missing_flags,
                population[concurrent],
                population[outcomes],
            ],
            axis=1,
        ).reset_index(drop=True)
        view[cat] = view[cat].astype("string")
        return view

    def create_impact_datasets(
        self, master: pd.DataFrame | None = None
    ) -> dict[str, pd.DataFrame]:
        """Save one impact view per cohort, treatment year, and aid source."""
        master = self.prepare_impact_master(
            master if master is not None else self.load_master()
        )
        output_dir = self.project_dir / self.cfg["impact_output_dir"]
        datasets: dict[str, pd.DataFrame] = {}
        roles: dict[str, dict[str, Any]] = {}

        for cohort_name, cohort in self.cfg["impact_cohorts"].items():
            for year, outcomes in cohort["outcomes"].items():
                aid_columns = self.aid_columns(year)
                # Enrolled in year k = came back for every year before it.
                enrolled = master["STUDENT_TYPE"].eq(cohort["type"])
                for prior in range(1, year):
                    enrolled &= master[f"RETAINED_YR{prior}"].eq(1)
                # Recipients-only comparison: controls received other aid that year.
                population = master[enrolled & master[aid_columns].eq(1).any(axis=1)]
                num = cohort["num"] + self.history_columns(year)
                prior_aid = [
                    column for prior in range(1, year) for column in self.aid_columns(prior)
                ]

                for treatment in aid_columns:
                    source = treatment.rsplit("_src_", 1)[1]
                    name = f"{cohort_name}_yr{year}_{source}"
                    concurrent = [c for c in aid_columns if c != treatment]
                    view = self.build_impact_view(
                        population, num, cohort["cat"], prior_aid,
                        treatment, concurrent, outcomes,
                    )
                    n_treated = int(view["treated"].sum())
                    if n_treated in (0, len(view)):
                        self.logger.warning(
                            "%s has no %s group; effect cannot be estimated",
                            name,
                            "treated" if n_treated == 0 else "control",
                        )
                    save(view, output_dir / f"impact_{name}.parquet")
                    datasets[name] = view
                    roles[name] = {
                        "student_type": cohort["type"],
                        "treatment_year": year,
                        "treatment": treatment,
                        "treatment_column": "treated",
                        "covariates": num + cohort["cat"] + prior_aid,
                        "missing_indicators": [f"{c}_missing" for c in num],
                        "concurrent_aid": concurrent,
                        "outcomes": outcomes,
                        "n": len(view),
                        "n_treated": n_treated,
                    }
                    self.log(f"{name:28} n {len(view):6}  treated {n_treated:6}")

        config_path = self.project_dir / self.cfg["impact_config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(roles, config_file, indent=2)
        return datasets

def main() -> None:
    # Prediction views (create_datasets) are kept but not run: the project
    # estimates aid effects, which only needs the impact views.
    ModelingDatasetCreation().create_impact_datasets()


if __name__ == "__main__":
    main()
