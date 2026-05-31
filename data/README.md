# Data sources

This directory holds the raw external trade data that feeds the
dissertation pipeline. Files are partitioned by source.

## ASK — Kosovo Agency of Statistics (active workbooks, May 2026)

Source: https://askdata.rks-gov.net/pxweb/en/ASKdata/ASKdata__External%20trade__Yearly%20indicators/
Downloaded: 2026-04

| File                                                       | ASK table                                  |
|------------------------------------------------------------|--------------------------------------------|
| `ask/ask_yearly_partner_trade_2010_2024.xlsx`              | Export and Import by partner country       |
| `ask/ask_yearly_hs_sections_trade_2010_2024.xlsx`          | Export and Import by sections HS           |
| `ask/ask_yearly_chapter_trade_2010_2024.xlsx`              | Export and Import by chapter               |
| `ask/ask_yearly_bec_trade_2010_2024.xlsx`                  | Export and Import by BEC classification    |
| `ask/ask_monthly_import_partner_2010M01_2026M03.xlsx`      | Import by partner country (monthly)        |
| `ask/ask_monthly_export_partner_2010M01_2026M03.xlsx`      | Export by partner country (monthly)        |

The pipeline reads these workbooks via `dissertation/src/data_pipeline.py`
(yearly partner + HS sections, for the canonical 1,695-row panel) and
via `dissertation/src/eda.py` (all six, for Phase 0 reconciliation and
the monthly Serbia event study). Default paths in source code resolve
to `data/ask/`.

### SHA-256 checksums

```
0e13be4bde15dd90d4392a579b5948685e44a94e505b4e8b3a81de2662b2e9eb  ask_yearly_partner_trade_2010_2024.xlsx
ab5d78c71dfb2c7169206479afdc3b0904d1c6877015c8ed7c418999a1dc22ab  ask_yearly_hs_sections_trade_2010_2024.xlsx
ef2e81879a1b92311b0ae3743aa01ac385186bfcb3d80e775b9c0444be5aa6af  ask_yearly_chapter_trade_2010_2024.xlsx
7a8fe26afec935512e9385d45f5dcc724277ffcec24fddc7e23a077cca5a6c44  ask_yearly_bec_trade_2010_2024.xlsx
794ab837fac0ff21d68b059b8c7cb3e26ed5bec461820dbd7e5f23b4484476f1  ask_monthly_import_partner_2010M01_2026M03.xlsx
db057b1142db4077f0ee4635467b2c7d1c63decdcc580316aa16f9a2b4945e5b  ask_monthly_export_partner_2010M01_2026M03.xlsx
```

## BACI — CEPII reference codes

Source: https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37
Release: V202501 (January 2025)

| File                                              | Purpose                                              |
|---------------------------------------------------|------------------------------------------------------|
| `baci/country_codes_V202501.csv`                  | BACI country – ISO2 – ISO3 lookup                    |
| `baci/product_codes_HS02_V202501.csv`             | HS chapter / heading codes and descriptions          |

These are reference tables; the active pipeline does not depend on
them. They are kept for cross-referencing trade-product descriptions
in the dissertation write-up.
