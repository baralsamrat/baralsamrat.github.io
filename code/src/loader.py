"""
loader.py
---------
Handles robust ingestion of all raw data files.
Each reader strips whitespace from column names and handles malformed headers
that are common in health data exports from DHIS2-style systems.

DATA_DIR resolution
-------------------
This module is imported from `code/src/loader.py`. Source CSVs live at the
repo root in `data/`. We resolve `DATA_DIR` relative to this file so the
pipeline can be invoked from any working directory (Quarto build, ad-hoc
REPL, or `python code/main.py` from repo root) without `cd` gymnastics.
The TYPHOID_DATA_DIR environment variable, if set, overrides the default.
"""

import os
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))               # code/src/
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))    # repo root
_DEFAULT_DATA_DIR = os.path.join(_REPO_ROOT, "data")


class DataLoader:
    """Centralised data loader with defensive header detection."""

    DATA_DIR = os.environ.get("TYPHOID_DATA_DIR", _DEFAULT_DATA_DIR)

    TYPHOID_FILES = ["typhoid_data_1.csv", "typhoid_data_2.csv"]
    CLIMATE_FILE  = "climate_data.csv"
    FLOOD_FILE    = "flood_data.csv"

    # ------------------------------------------------------------------ #
    #  Typhoid data                                                        #
    # ------------------------------------------------------------------ #
    def load_typhoid_data(self) -> pd.DataFrame:
        frames = []
        for fname in self.TYPHOID_FILES:
            path = os.path.join(self.DATA_DIR, fname)
            if not os.path.exists(path):
                continue
            # Try skipping 0, 1, or 2 leading rows to find real header
            for skip in range(3):
                try:
                    df = pd.read_csv(path, skiprows=skip, low_memory=False)
                    df.columns = [c.strip() for c in df.columns]
                    if "periodname" in df.columns:
                        frames.append(df)
                        break
                except Exception:
                    continue
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Climate data                                                        #
    # ------------------------------------------------------------------ #
    def load_climate_data(self) -> pd.DataFrame:
        path = os.path.join(self.DATA_DIR, self.CLIMATE_FILE)
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return df

    # ------------------------------------------------------------------ #
    #  Flood data                                                          #
    # ------------------------------------------------------------------ #
    def load_flood_data(self) -> pd.DataFrame:
        path = os.path.join(self.DATA_DIR, self.FLOOD_FILE)
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return df

    # ------------------------------------------------------------------ #
    #  Convenience wrapper                                                 #
    # ------------------------------------------------------------------ #
    def load_all(self) -> dict:
        return {
            "typhoid": self.load_typhoid_data(),
            "climate": self.load_climate_data(),
            "flood":   self.load_flood_data(),
        }
