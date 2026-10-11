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


def to_code(series: pd.Series, width: int | None = None) -> pd.Series:
    """Store a code as text: keeps nulls, drops '.0', optionally zero-pads."""
    if pd.api.types.is_float_dtype(series):
        series = series.astype("Int64")
    result = series.astype("string").str.strip()
    if width:
        result = result.str.zfill(width)
    return result


class MasterTableCreation:
    def __init__(
        self,
        config_path: str | Path | None = None,
        project_dir: str | Path | None = None,
    ) -> None:
        self.project_dir = resolve_project_dir(project_dir)
        self.cfg: dict[str, Any] = load_config("ml.yaml", self.project_dir, config_path)
        self.logger = logging.getLogger(__name__)
        self.aid_mismatch_ids: set[str] = set()

    def log(self, message: str) -> None:
        self.logger.info(message)

    def load_sources(self) -> dict[str, pd.DataFrame]:
        return load_sources(self.project_dir, self.logger)

    def build_aid_wide(self, aid: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
        aid = aid.copy()
        cohort = static.set_index("STUDENT_ID")["COHORT_YEAR"]
        aid["COHORT_YEAR_static"] = aid["STUDENT_ID"].map(cohort)
        n_missing = aid["COHORT_YEAR_static"].isna().sum()
        self.log(f"Aid rows whose student is not in static: {n_missing}")
        aid = aid[aid["COHORT_YEAR_static"].notna()].copy()

        self.aid_mismatch_ids = set(
            aid.loc[aid["COHORT_YEAR"] != aid["COHORT_YEAR_static"], "STUDENT_ID"]
        )
        aid["yr_num"] = (
            pd.to_numeric(aid["AID_YEAR"], errors="raise")
            - aid["COHORT_YEAR_static"].str[-4:].astype(int)
            + 1
        )

        lo, hi = self.cfg["aid_years"]
        assert (aid["yr_num"] >= lo).all(), "aid recorded before year 1"
        self.log(f"Aid rows after year {hi} dropped: {(aid['yr_num'] > hi).sum()}")
        aid = aid[aid["yr_num"].between(lo, hi)].copy()

        aid["group"] = (
            aid["FA_SOURCE_DESCR"]
            .str.lower()
            .str.replace(r"\W+", "_", regex=True)
            .str.strip("_")
        )

        wide = pd.crosstab(aid["STUDENT_ID"], [aid["yr_num"], aid["group"]]).clip(
            upper=1
        )
        wide.columns = [f"aid_yr{year}_src_{source}" for year, source in wide.columns]

        sources = self.cfg["aid_sources"]
        expected = [
            f"aid_yr{year}_src_{source}"
            for year in range(lo, hi + 1)
            for source in sources
        ]
        unexpected = set(wide.columns) - set(expected)
        assert not unexpected, f"Unexpected aid columns: {unexpected}"
        wide = wide.reindex(columns=expected, fill_value=0).astype("Int8")
        wide.index.name = "STUDENT_ID"
        wide = wide.reset_index()

        assert wide["STUDENT_ID"].is_unique
        return wide

    def prepare_term(self, term: pd.DataFrame) -> pd.DataFrame:
        term = term.drop(columns=self.cfg["term_drop_cols"]).copy()
        key = ["STUDENT_ID", "CF_BOE_TERM_ID"]
        n_before = len(term)
        term["_n_null"] = term.isna().sum(axis=1)
        term = (
            term.sort_values(key + ["_n_null"])
            .drop_duplicates(subset=key, keep="first")
            .drop(columns="_n_null")
        )
        self.log(f"Duplicate student+term rows removed: {n_before - len(term)}")
        assert not term.duplicated(subset=key).any()

        term = term.sort_values(key)
        acad_start = term["ACAD_YEAR"].str[:4].astype(int)
        n_back = acad_start.groupby(term["STUDENT_ID"]).diff().lt(0).sum()
        assert n_back == 0, f"{n_back} terms where ACAD_YEAR goes backward"
        return term

    def add_term_year(self, term: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
        term = term.copy()
        cohort = static.set_index("STUDENT_ID")["COHORT_YEAR"]
        term["COHORT_YEAR_static"] = term["STUDENT_ID"].map(cohort)
        assert term["COHORT_YEAR_static"].notna().all(), (
            "term students missing from static"
        )

        acad_end = term["ACAD_YEAR"].str[:4].astype(int) + 1
        cohort_end = term["COHORT_YEAR_static"].str[-4:].astype(int)
        term["term_yr"] = acad_end - cohort_end + 1
        self.log(f"Terms before year 1: {(term['term_yr'] < 1).sum()}")
        return term.drop(columns="COHORT_YEAR_static")

    def build_first_term(self, term: pd.DataFrame) -> pd.DataFrame:
        first = (
            term.sort_values(["STUDENT_ID", "CF_BOE_TERM_ID"])
            .drop_duplicates("STUDENT_ID", keep="first")[self.cfg["first_cols"]]
            .rename(
                columns={
                    "COHORT_YEAR": "COHORT_YEAR_term",
                    "CF_BOE_TERM_ID": "first_term_id",
                    "TERM_DESCR": "first_term_descr",
                    "ACAD_YEAR": "first_acad_year",
                }
            )
        )
        transfer_max = (
            term.groupby("STUDENT_ID")["Credits_Transfered"]
            .max()
            .rename("Credits_Transfered_Max")
            .reset_index()
        )
        first = first.merge(
            transfer_max, on="STUDENT_ID", how="left", validate="one_to_one"
        )

        assert first["STUDENT_ID"].is_unique
        assert len(first) == term["STUDENT_ID"].nunique()
        return first

    def build_year_summaries(self, term: pd.DataFrame) -> pd.DataFrame:
        lo, hi = self.cfg["term_years"]
        selected = (
            term[term["term_yr"].between(lo, hi)]
            .sort_values(["STUDENT_ID", "CF_BOE_TERM_ID"])
            .copy()
        )
        selected["cum_hrs_end"] = (
            selected["CUM_TOTHRS_EARN_BEGIN_TERM"] + selected["Credits_Earned_In_Term"]
        )

        def total(series: pd.Series) -> object:
            return series.sum(min_count=1)

        aggregated = (
            selected.groupby(["STUDENT_ID", "term_yr"])
            .agg(
                n_terms=("CF_BOE_TERM_ID", "nunique"),
                credits_enrolled=("CF_BOE_TERM_CRS_LD", total),
                credits_earned=("Credits_Earned_In_Term", total),
                end_cum_gpa=("CUM_GPA_END_TERM", "last"),
                end_ucf_gpa=("UCF_GPA_END_TERM", "last"),
                end_cum_hrs=("cum_hrs_end", "last"),
            )
            .unstack("term_yr")
        )
        if isinstance(aggregated, pd.Series):
            aggregated = aggregated.to_frame()
        by_year = aggregated
        by_year.columns = [f"yr{year}_{metric}" for metric, year in by_year.columns]

        metrics = [
            "n_terms",
            "credits_enrolled",
            "credits_earned",
            "end_cum_gpa",
            "end_ucf_gpa",
            "end_cum_hrs",
        ]
        expected = [
            f"yr{year}_{metric}" for year in range(lo, hi + 1) for metric in metrics
        ]
        by_year = by_year.reindex(columns=expected).reset_index()
        if "STUDENT_ID" not in by_year:
            by_year.insert(0, "STUDENT_ID", pd.Series(dtype="object"))

        assert by_year["STUDENT_ID"].is_unique
        return by_year

    def build_master(
        self,
        static: pd.DataFrame,
        aid_wide: pd.DataFrame,
        first_term: pd.DataFrame,
        by_year: pd.DataFrame,
    ) -> pd.DataFrame:
        master = static.merge(
            aid_wide, on="STUDENT_ID", how="inner", validate="one_to_one"
        )
        recipient_count = len(master)
        self.log(f"Recipients in master: {recipient_count}")
        master = master.merge(
            first_term, on="STUDENT_ID", how="left", validate="one_to_one"
        )
        master = master.merge(
            by_year, on="STUDENT_ID", how="left", validate="one_to_one"
        )

        assert len(master) == recipient_count, "a merge changed the row count"
        if self.cfg.get("expected_rows") is not None:
            assert recipient_count == self.cfg["expected_rows"], (
                f"expected {self.cfg['expected_rows']}, got {recipient_count}"
            )
        assert master["STUDENT_ID"].is_unique
        assert master["first_term_id"].notna().all(), (
            "some recipients have no term rows"
        )
        leftovers = [
            column
            for column in master.columns
            if isinstance(column, str) and column.endswith(("_x", "_y"))
        ]
        assert not leftovers, f"duplicated columns from a merge: {leftovers}"

        term_count_cols = [
            column
            for column in master.columns
            if isinstance(column, str) and column.endswith("_n_terms")
        ]
        master[term_count_cols] = master[term_count_cols].fillna(0)
        master["cohort_mismatch_term"] = (
            master["COHORT_YEAR"] != master["COHORT_YEAR_term"]
        )
        master["cohort_mismatch_aid"] = master["STUDENT_ID"].isin(self.aid_mismatch_ids)
        return master

    def post_merge_cleanup(self, master: pd.DataFrame) -> pd.DataFrame:
        master = master.copy()
        if self.cfg["sat_zero_is_missing"]:
            sat_columns = ["SAT_TOTAL", "SAT_VERB", "SAT_MATH"]
            master[sat_columns] = master[sat_columns].mask(master[sat_columns] == 0)

        master["re_entered"] = master["BOE_RECENT_TYPE_DATE"] != master["Entry_Date"]
        master = master.drop(columns=self.cfg["post_merge_drop_cols"])

        nulls_before = master.isna().sum()
        code_columns = master.select_dtypes(include=["object", "str"]).columns
        for column in code_columns:
            master[column] = to_code(master[column])
        master["CF_BOE_NM_IND"] = to_code(master["CF_BOE_NM_IND"])
        master["Entry_CIP"] = to_code(master["Entry_CIP"], width=6)

        grad_flag_cols: list[str] = self.cfg["grad_flag_cols"]
        flag_columns: list[str] = [
            "AA_AS",
            "RETAINED_YR1",
            "RETAINED_YR2",
            "RETAINED_YR3",
            "re_entered",
            "cohort_mismatch_term",
            "cohort_mismatch_aid",
            *grad_flag_cols,
            *[
                column
                for column in master.columns
                if isinstance(column, str)
                and column.endswith(("_FULL_TIME", "_PART_TIME"))
            ],
            *[
                column
                for column in master.columns
                if isinstance(column, str) and column.startswith("aid_yr")
            ],
        ]
        master[flag_columns] = master[flag_columns].astype("Int8")

        integer_columns = ["Entry_Age", "DEG_OUT_YR"] + [
            column
            for column in master.columns
            if isinstance(column, str) and column.endswith("_n_terms")
        ]
        master[integer_columns] = master[integer_columns].astype("Int16")
        master["DEGREE_TERM"] = master["DEGREE_TERM"].astype("Int64")

        changed = master.isna().sum() != nulls_before
        assert not changed.any(), (
            f"type conversion changed nulls in: {list(changed[changed].index)}"
        )
        output_columns: list[str] = self.cfg["output_columns"]
        missing_columns = [column for column in output_columns if column not in master]
        if missing_columns:
            raise KeyError(
                f"Configured output columns are missing from the merged dataset: "
                f"{missing_columns}"
            )
        return master.loc[:, output_columns].copy()

    def create_datasets(
        self, sources: dict[str, pd.DataFrame] | None = None
    ) -> pd.DataFrame:
        data = sources if sources is not None else self.load_sources()
        static = data["student_static"]
        aid_wide = self.build_aid_wide(data["financial_aid"], static)
        term = self.prepare_term(data["student_term"])
        term = self.add_term_year(term, static)
        first_term = self.build_first_term(term)
        by_year = self.build_year_summaries(term)
        master = self.build_master(static, aid_wide, first_term, by_year)
        master = self.post_merge_cleanup(master)

        output_path = self.project_dir / self.cfg["output_path"]
        save(master, output_path)
        return master

    def output_path(self) -> Path:
        return self.project_dir / self.cfg["output_path"]


def main() -> None:
    MasterTableCreation().create_datasets()


if __name__ == "__main__":
    main()
