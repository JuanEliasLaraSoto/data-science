# CLAUDE.md

Portfolio project by Juan Elías Lara Soto (double degree in Mathematics and Computer
Engineering, University of Málaga), aimed at quantitative risk consulting roles. It has
to show scoring models, model validation, Monte Carlo simulation and stress testing
done with the rigour of a bank's risk team.

## Working rules

- Code, comments, README and reports in **English**. Talk to the user in **Spanish**.
- Work phase by phase. **Before each phase**, give a 3–5 line plan and wait for the OK.
- At the end of each phase: run lint, mypy and tests, commit, and **stop**. Then
  summarise in Spanish what was done and *why* each methodological decision was taken,
  at the depth needed to defend it in a technical interview.
- Favour clarity and statistical rigour over complexity. No deep learning.
- Fixed seeds everywhere (`random_state=42`); results must be reproducible.
- Never invent results: every number in the README or reports comes from an executed
  script.

## Stack and quality gates

- Python 3.11+, managed with `uv` (`uv sync`, `uv run ...`). Lockfile `uv.lock` is committed.
- pandas, numpy, scikit-learn, statsmodels, scipy, matplotlib, seaborn. WoE/IV is an own,
  tested implementation in `woe.py`.
- `uv run ruff check . && uv run ruff format --check .`
- `uv run mypy` (`--strict` on `src/` and `scripts/`)
- `uv run pytest` (coverage on `src/` must stay >= 85 %)
- `make check` runs all of the above; CI runs the same on every push.
- Logic lives in `src/credit_risk_lab/`; notebooks only import it for exploration and figures.
- Modules beyond the original plan: `selection.py` (IV/correlation/VIF filters), `eda.py`,
  `plotting.py` (shared figure style, fixed CVD-safe palette), `reporting.py` (Markdown tables).
- Each phase script writes its tables, figures and a `summary.md` under `reports/`; numbers
  quoted anywhere else must come from those files.

## Data rules

- Lending Club accepted loans 2007–2018 (Kaggle `wordsforthewise/lending-club`).
  Target = 1 for `Charged Off`/`Default`, 0 for `Fully Paid`; drop unresolved loans
  (`Current`, `Late`, `In Grace Period`, ...).
- Drop every field known only after origination (payments, recoveries, `last_pymnt_*`, ...);
  keep the list and the reason for each field in code and in the README.
- Split: stratified random train/test on older vintages + out-of-time on the most recent
  vintages.
- Fallback: German Credit (OpenML `credit-g`) for phases 1–2 only (no dates, so no OOT).
- Macro: FRED public CSV (no API key), aggregated to quarterly.
- `data/raw` and `data/processed` are git-ignored; only `scripts/download_data.py` is versioned.

## Phases

0. Setup. 1. Scorecard. 2. Validation report (PDF). 3. Vasicek Monte Carlo portfolio.
4. ARIMAX macro stress testing. 5. Optional (R clustering, PySpark pipeline).
