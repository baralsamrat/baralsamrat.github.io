"""
processor.py
------------
Normalises district names, converts the Nepali Bikram Sambat (BS) calendar
to Gregorian (AD), handles outliers via IQR-clipping, and merges all
three datasets into a single panel DataFrame.

Calendar convention:
    AD year  =  BS year - 57   (Baisakh–Poush, i.e. April–December)
    AD year  =  BS year - 56   (Magh–Chaitra, i.e. January–March)
    This gives the correct AD month for each BS month.
"""

import re
import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────── #
#  Short, clean column aliases used throughout the project               #
# ────────────────────────────────────────────────────────────────────── #
COL_MAP = {
    "Min temperature (ERA5-Land)":      "temp_min",
    "Max air temperature (ERA5-Land)":  "temp_max",
    "Air temperature (ERA5-Land)":      "temp_mean",
    "Precipitation (CHIRPS)":           "precip",
    "Relative humidity (ERA5-Land)":    "humidity",
}


class DataProcessor:
    """Clean, merge, and lightly transform the three source datasets."""

    # Bikram Sambat month name → approximate Gregorian month number
    BS_MONTH_TO_AD = {
        "Baisakh": 4, "Jestha": 5,  "Asar":    6, "Shrawan": 7,
        "Bhadra":  8, "Ashwin": 9,  "Kartik": 10, "Mangsir": 11,
        "Poush":  12, "Magh":   1,  "Falgun":  2, "Chaitra":  3,
    }

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_district(name) -> str:
        """Strip leading numeric codes (e.g. '503 PYUTHAN' → 'PYUTHAN')."""
        if pd.isna(name):
            return "Unknown"
        return re.sub(r"^\d+\s+", "", str(name)).strip().upper()

    def _bs_to_ad(self, period_str: str):
        """
        Convert a BS month-string such as 'Asar 2072' to '2015-06'.
        Returns None if parsing fails.
        """
        try:
            parts = str(period_str).split()
            if len(parts) != 2:
                return None
            month_name, year_bs = parts[0], int(parts[1])
            ad_month = self.BS_MONTH_TO_AD.get(month_name)
            if ad_month is None:
                return None
            # Magh / Falgun / Chaitra fall in the *next* Gregorian year
            ad_year = year_bs - 57 if ad_month >= 4 else year_bs - 56
            return f"{ad_year}-{ad_month:02d}"
        except Exception:
            return None

    @staticmethod
    def _remove_outliers_iqr(series: pd.Series, factor: float = 3.0) -> pd.Series:
        """
        Winsorise a numeric series using IQR with a generous multiplier (3×)
        to catch only genuine extreme outliers while preserving real peaks.
        """
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        return series.clip(lower=lower, upper=upper)

    # ------------------------------------------------------------------ #
    #  Dataset-specific processors                                         #
    # ------------------------------------------------------------------ #
    def process_typhoid(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Rename the long column that contains 'Typhoid'
        typhoid_col = next(c for c in df.columns if "Typhoid" in c)
        df.rename(
            columns={
                typhoid_col:            "cases",
                "organisationunitname": "district",
                "periodname":           "period",
            },
            inplace=True,
        )

        # Clean case counts (DHIS2 exports use commas as thousands separator)
        df["cases"] = (
            df["cases"].astype(str).str.replace(",", "", regex=False).astype(float)
        )

        df["district"] = df["district"].map(self._normalize_district)
        df["date_ad"]  = df["period"].map(self._bs_to_ad)

        # Drop rows with unmappable dates or missing case data
        df = df.dropna(subset=["date_ad", "cases"])

        # Winsorise extreme case counts (province-level totals, etc.)
        df["cases"] = self._remove_outliers_iqr(df["cases"])

        return df[["district", "date_ad", "cases"]]

    def process_climate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.rename(columns=COL_MAP, inplace=True)
        df["district"] = df["district"].map(self._normalize_district)
        # period format: 201501 → 2015-01
        df["date_ad"] = df["period"].astype(str).str.replace(
            r"(\d{4})(\d{2})", r"\1-\2", regex=True
        )
        return df[["district", "date_ad", "temp_min", "temp_max",
                   "temp_mean", "precip", "humidity"]]

    def process_flood(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["district"] = df["district"].map(self._normalize_district)
        # period format: 20230615 → 2023-06  (daily events → monthly count)
        df["date_ad"] = df["period"].astype(str).str.replace(
            r"(\d{4})(\d{2})\d{2}", r"\1-\2", regex=True
        )
        monthly = (
            df.groupby(["district", "date_ad"])
            .size()
            .reset_index(name="flood_events")
        )
        return monthly

    # ------------------------------------------------------------------ #
    #  Merge                                                               #
    # ------------------------------------------------------------------ #
    def merge_datasets(
        self,
        typhoid_df: pd.DataFrame,
        climate_df: pd.DataFrame,
        flood_df:   pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Inner-join climate × typhoid (ensuring only overlapping districts and
        months are kept), then left-join flood data so that months without
        recorded floods are represented as zero events.
        """
        merged = pd.merge(
            climate_df, typhoid_df,
            on=["district", "date_ad"],
            how="inner",
        )
        merged = pd.merge(
            merged, flood_df,
            on=["district", "date_ad"],
            how="left",
        )
        merged["flood_events"] = merged["flood_events"].fillna(0)
        merged = merged.sort_values(["district", "date_ad"]).reset_index(drop=True)
        return merged
