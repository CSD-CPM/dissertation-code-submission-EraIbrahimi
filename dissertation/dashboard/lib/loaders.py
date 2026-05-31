"""Cached artefact loaders and path helpers shared by every dashboard page."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from src import config as cfg


def M(name: str) -> str:
    return str(cfg.METRICS / name)


def T(name: str) -> str:
    return str(cfg.TABLES / name)


def F(name: str) -> str:
    return str(cfg.FIGURES / name)


def P(name: str) -> str:
    return str(cfg.DATA_PROCESSED / name)


@st.cache_data
def load_csv(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_data
def load_json(path_str: str) -> dict:
    with open(path_str) as f:
        return json.load(f)


@st.cache_data
def load_npy(path_str: str) -> np.ndarray:
    return np.load(path_str)


@st.cache_data
def load_parquet(path_str: str) -> pd.DataFrame:
    return pd.read_parquet(path_str)
