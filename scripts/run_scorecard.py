"""Phase 1: build, benchmark and evaluate the application scorecard.

Every table and figure reported for Phase 1 is written by this script to
``reports/scorecard/<dataset>/`` (tables, figures and ``summary.md``). The fitted
scorecard and the scored samples go to ``data/processed/<dataset>/`` for Phase 2.

Examples
--------
Lending Club (main dataset)::

    uv run python scripts/run_scorecard.py

German Credit fallback (no dates, hence no out-of-time sample)::

    uv run python scripts/run_scorecard.py --dataset german-credit
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk_lab import data, eda, metrics
from credit_risk_lab.benchmark import fit_challengers
from credit_risk_lab.plotting import apply_style, save_figure
from credit_risk_lab.reporting import markdown_table, plot_binnings
from credit_risk_lab.scorecard import PointsScaling, Scorecard
from credit_risk_lab.selection import select_features
from credit_risk_lab.woe import WoEEncoder

PROJECT_ROOT = data.PROJECT_ROOT
logger = logging.getLogger("run_scorecard")

# Binning and selection settings (see woe.py and selection.py for the rationale).
MIN_BIN_FRAC = 0.05
MAX_BINS = 8
MIN_IV, MAX_IV = 0.02, 0.5
MAX_CORR, MAX_VIF = 0.6, 5.0
MAX_PVALUE = 0.05

SCORECARD = "Scorecard (WoE logit)"
PERFORMANCE_COLUMNS = [
    "n", "default_rate", "mean_pd", "auc", "gini", "gini_ci_low", "gini_ci_high", "ks", "brier",
]  # fmt: skip

#: Characteristics with IV > MAX_IV that were reviewed for leakage and kept, with
#: the conclusion of the review. Anything else above MAX_IV is dropped.
REVIEWED_HIGH_IV: dict[str, dict[str, str]] = {
    "lending-club": {},
    "german-credit": {
        "checking_status": "status of the applicant's checking account at application; "
        "known before the credit decision, so a strong but legitimate predictor",
    },
}


def load_dataset(name: str, raw_dir: Path, tables: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the clean modelling table and the cleaning log."""
    if name == "lending-club":
        path = raw_dir / "lending_club" / data.LENDING_CLUB_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `uv run python scripts/download_data.py lending-club`."
            )
        logger.info("Reading %s", path)
        raw = data.load_lending_club(path)
        data.maturity_by_issue_year(raw).to_csv(tables / "maturity_by_issue_year.csv")
        data.exclusion_table().to_csv(tables / "excluded_columns.csv", index=False)
        return data.clean_lending_club(raw)
    path = raw_dir / "german_credit" / "credit_g.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python scripts/download_data.py german-credit`."
        )
    return data.clean_german_credit(pd.read_csv(path))


def run(args: argparse.Namespace) -> None:
    """Execute the whole Phase 1 pipeline."""
    slug = args.dataset.replace("-", "_")
    out = args.out_dir / slug
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    apply_style()

    # ------------------------------------------------------------------ data & split
    clean, cleaning_log = load_dataset(args.dataset, args.raw_dir, tables)
    cleaning_log.to_csv(tables / "cleaning_log.csv", index=False)
    has_dates = data.ISSUE_DATE in clean.columns
    design = data.SampleDesign(
        dev_start=args.dev_start,
        dev_end=args.dev_end,
        oot_start=args.oot_start,
        oot_end=args.oot_end,
    )
    samples = data.split_sample(clean, design)
    features = [c for c in clean.columns if c not in data.META_COLUMNS]
    sample_sizes = pd.DataFrame(
        {
            "loans": {k: len(v) for k, v in samples.items()},
            "bads": {k: int(v[data.TARGET].sum()) for k, v in samples.items()},
        }
    )
    sample_sizes["default_rate"] = sample_sizes["bads"] / sample_sizes["loans"]
    sample_sizes.to_csv(tables / "sample_sizes.csv")
    logger.info("Samples:\n%s", sample_sizes)
    train = samples["train"]

    # ------------------------------------------------------------------ EDA
    missing = eda.missing_rates(train, features)
    missing.rename("missing_share").to_csv(tables / "missing_rates_train.csv")
    if has_dates:
        in_scope = pd.concat(samples.values())
        vintages = eda.default_rate_by_vintage(in_scope)
        vintages.to_csv(tables / "default_rate_by_vintage.csv")
        windows = {
            "Development": (
                str(pd.Period(design.dev_start, "Q")),
                str(pd.Period(design.dev_end, "Q")),
            ),
            "Out-of-time": (
                str(pd.Period(design.oot_start, "Q")),
                str(pd.Period(design.oot_end, "Q")),
            ),
        }
        ax = eda.plot_default_rate_by_vintage(vintages, windows)
        ax.set_title("Default rate by quarterly vintage (36-month loans)")
        save_figure(ax, figures / "default_rate_by_vintage.png")
        with_missing = [c for c in features if missing[c] > 0]
        if with_missing:
            by_year = eda.missing_by_year(in_scope, with_missing)
            by_year.to_csv(tables / "missing_by_issue_year.csv")
            ax = eda.plot_missing_by_year(by_year)
            ax.set_title("Share of missing values by issue year")
            save_figure(ax, figures / "missing_by_issue_year.png")

    # ------------------------------------------------------------------ WoE, IV, selection
    y_train = train[data.TARGET].to_numpy()
    encoder = WoEEncoder(min_bin_frac=MIN_BIN_FRAC, max_bins=MAX_BINS).fit(train[features], y_train)
    iv_table = encoder.iv_table()
    iv_table.to_csv(tables / "iv_table.csv", index=False)
    pd.concat(
        [b.table.assign(feature=f) for f, b in encoder.binnings_.items()], ignore_index=True
    ).to_csv(tables / "binning_tables.csv", index=False)

    woe_train = encoder.transform(train[features])
    selection = select_features(
        woe_train,
        iv_table.set_index("feature")["iv"],
        min_iv=MIN_IV,
        max_iv=MAX_IV,
        max_corr=MAX_CORR,
        max_vif=MAX_VIF,
        reviewed_high_iv=REVIEWED_HIGH_IV[args.dataset],
    )
    numeric_top = [f for f in selection.selected if pd.api.types.is_numeric_dtype(train[f])][:6]
    if numeric_top:
        fig = eda.plot_distributions(train, numeric_top)
        save_figure(fig, figures / "distributions_top_features.png")

    # ------------------------------------------------------------------ scorecard
    scaling = PointsScaling(base_score=600, base_odds=50, pdo=20)
    card = Scorecard.fit(
        train, y_train, encoder, selection.selected, scaling=scaling, max_pvalue=MAX_PVALUE
    )
    selection_log = pd.concat([selection.log, card.elimination_log], ignore_index=True)
    selection_log.to_csv(tables / "selection_log.csv", index=False)
    card.summary.to_csv(tables / "logit_coefficients.csv")
    card.points_table().to_csv(tables / "scorecard_points.csv", index=False)
    fig = plot_binnings({f: card.binnings[f] for f in card.features})
    save_figure(fig, figures / "scorecard_binnings.png")
    logger.info("Scorecard characteristics: %s", card.features)

    predictions: dict[str, dict[str, np.ndarray]] = {
        SCORECARD: {k: card.predict_pd(v) for k, v in samples.items()}
    }

    # ------------------------------------------------------------------ challengers
    challengers_info: list[dict[str, object]] = []
    if not args.skip_benchmark:
        logger.info("Tuning challengers (3-fold CV on train)...")
        challengers = fit_challengers(train[card.features], y_train)
        for name, result in challengers.items():
            predictions[name] = {k: result.predict_pd(v[card.features]) for k, v in samples.items()}
            challengers_info.append(
                {"model": name, "cv_auc": result.cv_auc, "best_params": result.best_params}
            )
        pd.DataFrame(challengers_info).to_csv(tables / "challenger_tuning.csv", index=False)

    # ------------------------------------------------------------------ performance
    rows = []
    for model, by_sample in predictions.items():
        perf = metrics.performance_table(
            {k: (samples[k][data.TARGET], p) for k, p in by_sample.items()}
        )
        rows.append(perf.assign(model=model).rename_axis("sample").reset_index())
    performance = pd.concat(rows, ignore_index=True)[["model", "sample", *PERFORMANCE_COLUMNS]]
    performance.to_csv(tables / "performance.csv", index=False)

    # Paired DeLong test: does each challenger discriminate better than the scorecard?
    comparisons = []
    reference = predictions[SCORECARD]
    for model, by_sample in predictions.items():
        if model == SCORECARD:
            continue
        for sample in (k for k in samples if k != "train"):
            test = metrics.delong_test(
                samples[sample][data.TARGET], by_sample[sample], reference[sample]
            )
            comparisons.append(
                {
                    "challenger": model,
                    "sample": sample,
                    "gini_challenger": 2 * test["auc_1"] - 1,
                    "gini_scorecard": 2 * test["auc_2"] - 1,
                    "gini_difference": 2 * (test["auc_1"] - test["auc_2"]),
                    "p_value": test["p_value"],
                }
            )
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(tables / "model_comparison.csv", index=False)

    holdout = "oot" if "oot" in samples else "test"
    card_pd = predictions[SCORECARD]
    card_samples = {k: (samples[k][data.TARGET], card_pd[k]) for k in samples}
    ax = metrics.plot_roc(card_samples)
    ax.set_title("Scorecard ROC by sample")
    save_figure(ax, figures / "roc_by_sample.png")
    ax = metrics.plot_calibration(card_samples)
    ax.set_title("Scorecard calibration by sample (PD deciles)")
    save_figure(ax, figures / "calibration_by_sample.png")
    by_model = {m: (samples[holdout][data.TARGET], p[holdout]) for m, p in predictions.items()}
    ax = metrics.plot_roc(by_model)
    ax.set_title(f"ROC by model ({holdout} sample)")
    save_figure(ax, figures / "roc_by_model.png")
    ax = metrics.plot_calibration(by_model)
    ax.set_title(f"Calibration by model ({holdout} sample, PD deciles)")
    save_figure(ax, figures / "calibration_by_model.png")
    ax = metrics.plot_ks_curve(samples[holdout][data.TARGET], card.score(samples[holdout]))
    ax.set_title(f"Cumulative score distributions ({holdout} sample)")
    save_figure(ax, figures / "ks_curve.png")

    # ------------------------------------------------------------------ artefacts for Phase 2
    processed = args.processed_dir / slug
    processed.mkdir(parents=True, exist_ok=True)
    with (processed / "scorecard.pkl").open("wb") as fh:
        pickle.dump(card, fh)
    scored = []
    for name, frame in samples.items():
        meta = [c for c in data.META_COLUMNS if c in frame.columns]
        part = frame[meta + card.features].copy()
        part["sample"] = name
        part["pd"] = card_pd[name]
        part["score"] = card.score(frame)
        scored.append(part)
    scored_frame = pd.concat(scored, ignore_index=True)
    if data.VINTAGE in scored_frame.columns:
        scored_frame[data.VINTAGE] = scored_frame[data.VINTAGE].astype(str)
    scored_frame.to_parquet(processed / "scored_samples.parquet", index=False)

    write_summary(
        out / "summary.md",
        dataset=args.dataset,
        design=asdict(design) if has_dates else None,
        cleaning_log=cleaning_log,
        sample_sizes=sample_sizes,
        iv_table=iv_table,
        selection_log=selection_log,
        card=card,
        challengers=pd.DataFrame(challengers_info),
        performance=performance,
        comparison=comparison,
    )
    logger.info("Done. Outputs in %s", out)


def write_summary(
    path: Path,
    *,
    dataset: str,
    design: dict[str, object] | None,
    cleaning_log: pd.DataFrame,
    sample_sizes: pd.DataFrame,
    iv_table: pd.DataFrame,
    selection_log: pd.DataFrame,
    card: Scorecard,
    challengers: pd.DataFrame,
    performance: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """Write the Markdown summary of the run (every number comes from this run)."""
    perf = performance.set_index(["model", "sample"])[PERFORMANCE_COLUMNS]
    coefs = card.summary[["coef", "std_err", "z", "p_value"]]
    parts = [
        f"# Phase 1 scorecard: {dataset}",
        "_Generated by `scripts/run_scorecard.py`; do not edit by hand._",
        "## Sample",
        "Sample design: "
        + (
            ", ".join(f"{k} = {v}" for k, v in design.items())
            if design
            else "random split only (no dates)"
        ),
        markdown_table(cleaning_log, index=False),
        markdown_table(sample_sizes, float_format=".4f"),
        "## Characteristic screening",
        f"{len(iv_table)} candidate characteristics; "
        f"{(iv_table['iv'] >= MIN_IV).sum()} with IV >= {MIN_IV}; "
        f"{(iv_table['iv'] > MAX_IV).sum()} with IV > {MAX_IV} (suspicious).",
        markdown_table(iv_table.head(25), float_format=".4f", index=False),
        "### Dropped or flagged during selection",
        markdown_table(selection_log, index=False) if len(selection_log) else "None.",
        "## Logistic regression on WoE",
        f"{len(card.features)} characteristics. "
        + ", ".join(f"{k} = {v:,.4g}" for k, v in card.fit_stats.items()),
        markdown_table(coefs, float_format=".4g"),
        "## Challengers",
        markdown_table(challengers, float_format=".4f", index=False)
        if len(challengers)
        else "Skipped.",
        "## Performance",
        "Gini 95% confidence interval from the DeLong variance of the AUC.",
        markdown_table(perf, float_format=".4f"),
        "## Challenger vs scorecard (paired DeLong test)",
        markdown_table(comparison, float_format=".4f", index=False)
        if len(comparison)
        else "Skipped.",
    ]
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the pipeline; return a process exit code."""
    parser = argparse.ArgumentParser(description="Phase 1: application scorecard.")
    parser.add_argument(
        "--dataset", choices=["lending-club", "german-credit"], default="lending-club"
    )
    parser.add_argument("--raw-dir", type=Path, default=data.RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=data.PROCESSED_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "scorecard")
    defaults = data.SampleDesign()
    parser.add_argument("--dev-start", default=defaults.dev_start)
    parser.add_argument("--dev-end", default=defaults.dev_end)
    parser.add_argument("--oot-start", default=defaults.oot_start)
    parser.add_argument("--oot-end", default=defaults.oot_end)
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip the challenger models.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(args)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
