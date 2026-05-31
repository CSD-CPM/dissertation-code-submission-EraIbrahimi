# Kosovo Trade Analysis

MSc dissertation **AI-Driven Trade Policy Analysis for Small Open Economies — A
Study of Kosovo** (University of York Europe Campus / City College Thessaloniki;
author Era Ibrahimi). An interactive Streamlit dashboard and the reproducible
pipeline behind it.

**Live demo:** https://eraibrahimi-kosovo-trade-analysis.hf.space — hosted on
Hugging Face Spaces (free tier; if it has been idle it takes ~30–60 s to wake).

## Context

The project analyses **Kosovo's bilateral imports and exports** across a panel of
113 trade partners over 2010–2024 (n = 1,695 partner-year observations,
balanced). The empirical case study is **Serbia (XS)**: Kosovo's unilateral 100 %
tariff on Serbian goods, in force November 2018 – April 2020.

Three kinds of quantity are kept strictly distinct:

- **Descriptive** — the structure of Kosovo's trade (scale, deficit,
concentration, partner and sector composition).
- **Predictive (ŷ)** — XGBoost against a PPML baseline and a persistence floor,
interpreted with SHAP. SHAP values are *predictive contributions*, not effects.
- **Causal (β̂)** — a PPML difference-in-differences estimate of the  
tariff-attributable change. Serbia is treated; Albania, North Macedonia, and  
Montenegro (AL / MK / ME) are the controls.  
Results are reported as **suggestive**: pre-trends are not parallel and the 2014
placebo is non-zero, so the 1,000-draw stratified partner-pairs bootstrap
confidence interval — not the cluster-robust SE (only G = 4 clusters) — is the
canonical inference statement.

The central finding is an **asymmetry**: imports from Serbia collapse around the
2018 tariff (β ≈ −4.09, about −98 %) while exports show no comparable 2019 break
(β ≈ 0). Dashboard scenarios and the simulator are **conditional predictions under
manipulated inputs**, not policy effects.

## The dashboard

Five pages, in a descriptive → technical narrative (EDA → feature engineering →
prediction & modelling → simulation):

1. **Trade Landscape** — scale and balance, partners, concentration and sparsity, sectoral composition, and the Serbia tariff anatomy.
2. **Feature Engineering** — the twelve model inputs, how they are built, and the ablation layers.
3. **Causal** — β_DiD with bootstrap CI, safeguards, and the import-vs-export asymmetry.
4. **Predictive** — XGBoost vs PPML, a persistence baseline, SHAP, ablation, Diebold–Mariano.
5. **Simulator** — interactive what-if predictions (partner GDP, last-year trade).

The Causal and Predictive pages — and the sectoral section of Trade Landscape — carry an Imports / Exports toggle. The
dashboard does **no live training and no network calls** — it reads committed
artefacts.

## Requirements

- Python 3.12 (3.11+)
- ~2 GB disk, no GPU

## Run the dashboard

```bash
git clone https://github.com/EraIbrahimi/ai-trade-policy-analyzer-kosovo.git
cd ai-trade-policy-analyzer-kosovo/dissertation
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/app.py                          # serves http://localhost:8501
```

The processed panels, models, tables, metrics, and figures are committed, so a
fresh clone runs end to end without rebuilding anything. The Simulator runs
inference on the saved booster (partner-GDP and last-year-trade levers) — no
retraining required.

## Reproduce / retrain

Deterministic (`seed = 42` for numpy, Optuna, the bootstrap RNG, and XGBoost);
rebuilds `outputs/`, `data/processed/`, and `models/` from `data/raw/`. Run from
`dissertation/`:

```bash
python -m src.run_phase0      # descriptive EDA tables + figures
python -m src.run_phase1      # panels + PPML fits
python -m src.run_phase1_5    # export-side spillover DiD
python -m src.run_phase2      # XGBoost + Optuna (imports)
python -m src.run_phase2_5    # XGBoost (exports) + persistence baseline
python -m src.run_phase3      # SHAP + 4-layer ablation + Diebold–Mariano
python -m src.run_phase4      # DiD safeguards + 1,000-draw partner-pairs bootstrap
python -m src.run_phase5      # scenario artefacts
```

Each command is idempotent. Retraining the booster is `run_phase2` /
`run_phase2_5`; they overwrite `models/xgb_best*.joblib`, picked up on the next
dashboard launch.

## Test

```bash
cd dissertation && pytest        # artefact integrity, simulator, dashboard data contracts (~1s)
```

## Layout

```
dissertation/
├── dashboard/   app.py, pages/ (five pages), lib/ (loaders, theme, flow toggle)
├── src/         config, data_pipeline, features, ppml, xgb_model, interpret,
│                did_safeguards, scenarios, run_phase*
├── data/        raw/ (ASK, WDI, CEPII inputs), processed/ (parquet panels)
├── models/      saved boosters + SHAP values
├── outputs/     tables/, metrics/, figures/
├── tests/       pytest suite
└── requirements.txt
```

## Interpretation rules

Enforced consistently across the code, the dashboard prose, and every caption:

- **"Effect"** refers only to the DiD coefficient β — never to XGBoost output,
SHAP values, or scenarios.
- **DiD prose is hedged** ("the point estimate indicates", "consistent with",
"suggestive"); "the tariff caused" is not used.
- **SHAP values** are predictive contributions in the booster's log1p output
space; they are not back-transformed via `expm1`.
- **Scenarios and simulator outputs** are conditional predictions under
manipulated inputs. The Serbia-tariff scenario is a binary proxy (the booster
cannot distinguish 20 % from 100 %); the hypothetical Turkey FTA toggles
`cefta_member` as a proxy for an FTA-like agreement (Turkey is not a CEFTA
member, and this is not a forecast).

