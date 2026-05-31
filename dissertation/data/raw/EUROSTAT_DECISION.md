# Eurostat COMEXT HS2 — acquisition decision

**Decision: OPTIONAL (local download only)**

Date: 2026-04-19
Dataset: DS-045409 (Eurostat COMEXT, HS2 chapters, PARTNER=XK)
Time budget: 4 hours (hard cap per Final Dissertation Plan §7.1 Task C)

## Reason

The automated acquisition path cannot reach Eurostat in this project's
runtime environment. The following endpoints are not reachable without
manual intervention:

- https://ec.europa.eu/eurostat/databrowser/
- https://www.cepii.fr/
- https://api.worldbank.org/
- PyPI (files.pythonhosted.org, pypi.org)
- https://en.wikipedia.org/

The 4-hour time budget applies to wall-clock acquisition work on a local
machine.

## Consequences if not acquired

- Pillar 2 (Sector) relies on ASK tab04 + Gap Institute 2019, per §6.
- Dashboard Page 2's "Eurostat section" will be hidden (conditional rendering).
- No bilateral × sector diversion claims are made in Chapter 5.

## If acquired locally

See `DOWNLOAD_CHECKLIST.md § E` for exact filter settings. Place the
cleaned CSV at `data/raw/eurostat_comext_hs2.csv` with columns:
`partner, reporter, hs2, year, flow, value_eur`. The pipeline
auto-detects its presence and enables the bilateral × sector charts
without any code changes.
