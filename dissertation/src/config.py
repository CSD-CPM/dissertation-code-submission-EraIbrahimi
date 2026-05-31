"""Single source of truth for paths, constants, feature lists.

Bound by the implementation plan (Step 2) and the Final Dissertation Plan.
"""
from pathlib import Path

SEED = 42

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
METRICS = OUTPUTS / "metrics"

for p in (DATA_PROCESSED, MODELS, FIGURES, TABLES, METRICS):
    p.mkdir(parents=True, exist_ok=True)

# --- Panel filter ------------------------------------------------------------
# tab02 contains 252 partner codes (after dropping the ZZ aggregate).
# The plan's stated n = 1,695 corresponds to the top-113 partners by cumulative
# imports 2010-2024. This filter is documented in §11.8 of the implementation
# plan and applied in data_pipeline.py.
TOP_N_PARTNERS = 113
YEARS = list(range(2010, 2025))   # 15 years 2010..2024 inclusive
N_YEARS = len(YEARS)
EXPECTED_N = TOP_N_PARTNERS * N_YEARS  # 1,695

# --- CEFTA membership (Plan §6, Pillar 1 cefta_member feature) ---------------
CEFTA_MEMBERS_ISO2 = {"AL", "MK", "ME", "BA", "XS", "MD"}  # XS = Serbia in tab02 codes

# --- DiD design (Plan §6 Pillar 3) -------------------------------------------
DID_TREATED_ISO2 = "XS"
DID_CONTROL_ISO2 = ["AL", "MK", "ME"]
DID_TREATMENT_YEARS_PRIMARY = [2019]            # locked: primary treatment is 2019 only
DID_TREATMENT_YEARS_SENSITIVITY = [2018, 2019, 2020]

# --- Predictive serbia_tariff feature (Plan §6 Pillar 1) ---------------------
# Distinct from DID_TREATMENT_YEARS_PRIMARY: the predictive feature spans the
# full tariff window {2018,2019,2020} per plan; the DiD treatment indicator is
# 2019 only. Two different variables, two different specifications.
SERBIA_TARIFF_FEATURE_YEARS = {2018, 2019, 2020}

# --- Policy event years ------------------------------------------------------
SAA_IN_FORCE_FROM = 2016
COVID_YEAR = 2020

# --- EU members (for partner-conditional saa_in_force, §11.6) ----------------
EU_ISO2 = {
    "AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","GR","HU","IE",
    "IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE",
    # GB included for pre-Brexit years; SAA still applies to bilateral coverage
    "GB",
}

# --- Feature ordering (locked, used by every model and SHAP plot) ------------
FEATURE_ORDER = [
    "ln_partner_gdp",
    "ln_kosovo_gdp",
    "ln_distance",
    "contiguity",
    "common_language",
    "serbia_tariff",
    "saa_in_force",
    "covid",
    "cefta_member",
    "lagged_imports_log1p",
    "year_trend",
    "partner_import_share_lag",
]

# Ablation layers (§3 ablation table of the implementation plan)
ABLATION_LAYERS = {
    "L1_structural": ["ln_distance", "contiguity", "common_language"],
    "L2_policy":     ["ln_distance", "contiguity", "common_language",
                      "serbia_tariff", "saa_in_force", "covid", "cefta_member"],
    "L3_macro":      ["ln_distance", "contiguity", "common_language",
                      "serbia_tariff", "saa_in_force", "covid", "cefta_member",
                      "ln_partner_gdp", "ln_kosovo_gdp", "year_trend"],
    "L4_lagged":     FEATURE_ORDER,  # all 12
}

# --- PPML-Predictive regressor list (§4.1 of plan) ---------------------------
# Drops ln_kosovo_gdp due to perfect collinearity with year_trend in the
# panel (both vary only by year). XGBoost retains all 12 (§11.3).
PPML_PREDICTIVE_REGRESSORS = [
    "ln_partner_gdp",
    "saa_in_force",
    "serbia_tariff",
    "covid",
    "year_trend",
]

# --- CV configuration (§5.4 of plan) ----------------------------------------
# CV folds are now FIVE expanding-window splits — the (2022, 2023) fold was
# removed because the holdout already includes 2023. See PROTOCOL_FREEZE.md §4.
CV_FOLDS = [(2017, 2018), (2018, 2019), (2019, 2020),
            (2020, 2021), (2021, 2022)]
CV_HOLDOUT = (2022, [2023, 2024])  # train through 2022, test 2023 and 2024 — both genuinely untouched

# --- Bootstrap (locked: N = 1000 partners-with-replacement) ------------------
BOOTSTRAP_N = 1000

# --- Optuna -----------------------------------------------------------------
OPTUNA_TRIALS = 100
OPTUNA_SEARCH_SPACE = {
    "max_depth":        [3, 4, 5, 6],
    "n_estimators":     [100, 300, 500],
    "learning_rate":    [0.01, 0.05, 0.10],
    "reg_alpha":        [0.1, 1.0, 5.0],
    "reg_lambda":       [1.0, 3.0, 5.0],
    "min_child_weight": [5, 10, 20],
}

# --- WDI indicators (Plan §7.1 Task A) --------------------------------------
WDI_INDICATORS = {
    "NY.GDP.MKTP.CD":     "gdp_usd_current",
    "NY.GDP.MKTP.KD.ZG":  "gdp_growth_pct",
    "SP.POP.TOTL":        "population",
    "BX.KLT.DINV.WD.GD.ZS": "fdi_pct_gdp",
    "FP.CPI.TOTL.ZG":     "inflation_pct",
}
