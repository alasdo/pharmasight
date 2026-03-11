# PharmaSight

**Multi-source intelligence fusion for pharmaceutical demand forecasting**

PharmaSight combines 15 public data sources — spanning drug demand, supply chain disruptions, disease surveillance, FDA regulation, and news sentiment — into a unified ML forecasting system with a live operational dashboard.

This project targets two outputs: a **publishable preprint** (arXiv/medRxiv) presenting an ablation study across signal types, and a **production-grade ML pipeline** demonstrating end-to-end data engineering, modelling, and deployment.

---

## The Problem

Drug shortages in the US reached record highs in 2024. Essential medications — cancer treatments, antibiotics, anaesthetics — are routinely unavailable, directly impacting patient care. Yet the pharmaceutical industry still forecasts demand using surprisingly basic methods: last quarter's sales plus a manual adjustment.

Meanwhile, critical signals that shape drug demand are publicly available but ignored by existing forecasting systems:

- **Supply disruptions** — FDA-reported shortages, manufacturing delays, and recalls redistribute demand across therapeutic alternatives
- **Regulatory changes** — new generic approvals collapse branded volume by 80%+; policy changes in Medicaid reimbursement shift prescribing patterns overnight
- **Disease surveillance** — flu outbreaks drive antiviral and antibiotic demand with predictable seasonal patterns
- **News and sentiment** — safety scares and media coverage shift prescribing behaviour before any formal data reflects it
- **Patent cliffs** — expiry dates are known years in advance, yet their demand impact is rarely modelled explicitly

PharmaSight asks: **what if we combined all of these signals into a single forecasting system, and measured which ones actually matter?**

---

## Research Questions

| # | Question |
|---|---------|
| **RQ1** | Does incorporating multi-source heterogeneous data significantly improve pharmaceutical demand forecasting compared to historical demand alone? |
| **RQ2** | What is the relative predictive contribution of structured supply-side signals versus NLP-derived text signals? |
| **RQ3** | Can regulatory documents from the Federal Register serve as leading indicators for demand shifts, and at what optimal lead time? |
| **RQ4** | How do supply chain disruption events propagate through therapeutic substitution networks? |
| **RQ5** | How does model performance vary across therapeutic classes, and which classes benefit most from which signal types? |

---

## Data Architecture

PharmaSight ingests data from **15 public sources** across 5 categories, all joining through the National Drug Code (NDC) as the universal identifier.

### Structured Sources

| Source | Role | Granularity | Records |
|--------|------|-------------|---------|
| **Medicaid SDUD** | Primary demand signal | State × Drug × Quarter | 25.3M rows |
| **CDC FluView** | Disease driver | State × Week | 1997–present |
| **FDA Drug Shortages** | Supply disruption events | Drug × Event | Continuous |
| **Drugs@FDA** | Approval history & generic entries | Drug Application | 50,859 products |
| **Orange Book** | Patent expiry & therapeutic equivalence | Drug Product | 47,780 products |
| **OpenFDA FAERS** | Adverse event reports | Drug × Event × Quarter | Quarterly |
| **FDA Recalls** | Recall enforcement reports | Drug × Event | Continuous |

### Unstructured Sources — Regulation

| Source | Role | Access |
|--------|------|--------|
| **Federal Register API** | Proposed and final rules affecting pharma | REST API (no key) |
| **Regulations.gov API** | Rulemaking dockets and comment volumes | REST API (free key) |
| **FDA Guidance Documents** | Draft and final policy guidance | RSS |

### Unstructured Sources — News & Sentiment

| Source | Role | Access |
|--------|------|--------|
| **FDA Press Releases & RSS** | Official announcements and safety alerts | RSS |
| **Drugs.com RSS** | Pre-categorised pharma news | RSS |
| **NewsAPI** | Mainstream media coverage | REST API (free tier) |
| **Reddit** | Professional and patient discussion | REST API (free) |

### Pipeline Intelligence

| Source | Role | Access |
|--------|------|--------|
| **ClinicalTrials.gov** | Phase III completions as future market signals | REST API (no key) |

---

## Star Schema

All sources converge into a star schema optimised for analytical queries and ML feature engineering:

```
                  ┌─────────────┐
                  │ dim_product  │
                  │ (47,780)     │
                  └──────┬──────┘
                         │ ndc
    ┌──────────────┐     │     ┌───────────────┐
    │ feat_disease  ├─────┤     │ feat_supply    │
    │ (ILI rates)  │     │     │ (shortages,    │
    └──────────────┘     │     │  approvals)    │
                   ┌─────┴─────┐└───────────────┘
                   │           │
                   │   fact    │
                   │  demand   │
                   │ (18.7M)   │
                   │           │
                   └─────┬─────┘
    ┌──────────────┐     │     ┌───────────────┐
    │feat_regulation├─────┤     │ feat_safety    │
    │ (Fed Register)│     │     │ (FAERS,recalls)│
    └──────────────┘     │     └───────────────┘
                         │ state
                  ┌──────┴──────┐
                  │dim_geography │
                  │ (54 states)  │
                  └─────────────┘
```

---

## Methodology

### Model Comparison

| Model | Type | Purpose |
|-------|------|---------|
| Seasonal Naive | Baseline | Same quarter last year |
| LightGBM | Gradient Boosting | Strong tabular baseline |
| XGBoost | Gradient Boosting | Alternative ensemble |
| Temporal Fusion Transformer | Neural (Attention) | Multi-horizon with interpretability |
| N-BEATS | Neural | Pure time series |
| N-HiTS | Neural | Hierarchical interpolation |

### Ablation Study

The core research contribution is an ablation study measuring the marginal value of each signal layer:

| Config | Features |
|--------|----------|
| **A** (Baseline) | Demand lags + calendar features only |
| **B** (+ Structured) | A + shortage / approval / patent / disease / safety |
| **C** (+ News NLP) | B + news sentiment + social sentiment |
| **D** (+ Regulation NLP) | B + Federal Register + guidance + rulemaking |
| **E** (Full Fusion) | All features combined |

### NLP Pipeline

Unstructured text from regulatory filings, news, and social media is processed through:

1. **Named Entity Recognition** — SciSpacy for biomedical entity extraction
2. **Entity Linking** — mapping drug mentions to NDC / therapeutic class
3. **Event Classification** — categorising text as pricing, manufacturing, access, safety, or scheduling
4. **Sentiment Scoring** — domain-adapted transformer model
5. **Temporal Aggregation** — quarterly features per drug for model input

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Core Language | Python 3.12 |
| Data Processing | Pandas, Polars, DuckDB, PyArrow |
| ML — Tabular | LightGBM, XGBoost |
| ML — Neural | PyTorch, pytorch-forecasting (TFT, N-BEATS) |
| NLP | SciSpacy, HuggingFace Transformers |
| Experiment Tracking | MLflow |
| API Serving | FastAPI |
| Containerisation | Docker |
| Monitoring | Evidently AI |
| Dashboard | React |
| Storage | Parquet (columnar) |

---

## Project Structure

```
pharmasight/
├── config/
│   └── sources.yaml              # Source URLs, API params, schedules
├── src/
│   ├── extract/                   # One module per data source
│   │   ├── medicaid_sdud.py       # Medicaid State Drug Utilization Data
│   │   ├── drugs_at_fda.py        # FDA drug approvals & generic entries
│   │   ├── orange_book.py         # Patents & therapeutic equivalence
│   │   ├── fda_shortages.py       # Drug shortage reports
│   │   ├── fda_recalls.py         # Recall enforcement reports
│   │   ├── openfda_faers.py       # Adverse event reports
│   │   ├── cdc_fluview.py         # Influenza surveillance
│   │   ├── federal_register.py    # Federal regulation text
│   │   ├── regulations_gov.py     # Rulemaking dockets
│   │   ├── clinical_trials.py     # Clinical trial registry
│   │   ├── newsapi.py             # Mainstream media articles
│   │   └── reddit.py              # Social discussion
│   ├── transform/
│   │   ├── ndc_harmonise.py       # NDC format standardisation
│   │   ├── clean_sdud.py          # SDUD cleaning & validation
│   │   ├── build_dimensions.py    # Product & geography dimensions
│   │   ├── build_facts.py         # Core demand fact table
│   │   └── feature_eng.py         # Feature engineering pipeline
│   ├── validate/
│   │   └── contracts.py           # Pandera data contracts
│   └── utils/
│       └── api_client.py          # Rate-limited HTTP client
├── data/
│   ├── raw/                       # Untouched downloads (gitignored)
│   ├── validated/                 # After schema checks (gitignored)
│   └── processed/                 # Star schema tables (gitignored)
├── notebooks/                     # EDA and analysis
├── tests/                         # pytest suite
├── requirements.txt
├── Makefile
└── README.md
```

---

## Current Progress

- [x] Project structure and environment setup
- [x] Medicaid SDUD extraction (2019–2023, 25.3M rows, 70,592 NDCs)
- [x] NDC harmonisation utility (100% match rate)
- [x] Drugs@FDA extraction (50,859 products dating to 1939)
- [x] Orange Book extraction (47,780 products + 20,174 patents)
- [x] Product dimension table (71% enrichment via two-stage name matching)
- [x] Geography dimension table (54 states/territories with HHS regions)
- [x] Fact demand table (18.7M rows, $872B in reimbursements)
- [ ] FDA Shortages & Recalls extraction
- [ ] FAERS adverse event extraction
- [ ] CDC FluView extraction
- [ ] Federal Register & regulation text extraction
- [ ] News & social sentiment extraction
- [ ] NLP pipeline (NER, sentiment, event classification)
- [ ] Feature engineering
- [ ] Model training & ablation study
- [ ] FastAPI serving & Docker
- [ ] React dashboard
- [ ] Monitoring & drift detection
- [ ] Preprint

---

## Getting Started

### Prerequisites

- Python 3.12+
- 10GB+ disk space (for raw data downloads)

### Setup

```bash
git clone https://github.com/alasdo/pharmasight.git
cd pharmasight
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Extract Data

```bash
# Download Medicaid SDUD (2019-2023) — ~2.4GB
python -m src.extract.medicaid_sdud extract

# Download Drugs@FDA bulk data
python -m src.extract.drugs_at_fda extract

# Download Orange Book
python -m src.extract.orange_book extract
```

### Build Star Schema

```bash
# Clean and validate SDUD
python -m src.transform.clean_sdud

# Build dimension and fact tables
python -m src.transform.build_dimensions all
python -m src.transform.build_facts
```

### Verify

```bash
python -m src.extract.medicaid_sdud verify
python -m src.extract.drugs_at_fda verify
python -m src.extract.orange_book verify
```

---

## Data Sources & Licensing

All data sources used in this project are **publicly available** and free to access:

- Medicaid SDUD — US public domain
- FDA data (Drugs@FDA, Orange Book, Shortages, FAERS, Recalls) — US public domain via [openFDA](https://open.fda.gov)
- CDC FluView — US public domain via [Delphi Epidata API](https://cmu-delphi.github.io/delphi-epidata/)
- Federal Register — US public domain via [federalregister.gov API](https://www.federalregister.gov/developers/documentation/api/v1)
- ClinicalTrials.gov — US public domain
- NewsAPI — [newsapi.org](https://newsapi.org) (free tier, API key required)
- Reddit — [Reddit API](https://www.reddit.com/dev/api/) (free, OAuth required)

---

## Author

**Anas Lasri Doukkali** — Data Scientist at Amgen | PhD Mathematics (St Andrews) | BSc Mathematics (Imperial College London)

- Portfolio: [anaslasri.com](https://anaslasri.com)
- LinkedIn: [linkedin.com/in/anas-lasri-doukkali](https://www.linkedin.com/in/anas-lasri-doukkali/)
- Publication: [Agent-Based Modelling of Bladder Infections](https://www.frontiersin.org/articles/10.3389/fams.2023.1090334/full) — Frontiers in Applied Mathematics and Statistics, 2023

---

## License

This project is open source under the [MIT License](LICENSE).

*Note: This project is for research and educational purposes. The forecasting system is not intended for clinical decision-making. Always consult healthcare professionals regarding pharmaceutical supply decisions.*
