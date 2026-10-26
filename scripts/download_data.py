"""Download the raw datasets used by credit-risk-lab into ``data/raw``.

Examples
--------
Default sources (Lending Club + FRED)::

    uv run python scripts/download_data.py

Only the German Credit fallback::

    uv run python scripts/download_data.py german-credit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from credit_risk_lab import data

SOURCES = ("lending-club", "fred", "german-credit")
DEFAULT_SOURCES = ("lending-club", "fred")


def main(argv: list[str] | None = None) -> int:
    """Run the downloader and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    # No ``choices=`` here: on Python 3.11, argparse rejects an empty ``nargs="*"``
    # positional that has choices, so the names are validated below instead.
    parser.add_argument(
        "sources",
        nargs="*",
        metavar="SOURCE",
        help=f"One of {', '.join([*SOURCES, 'all'])} (default: {' '.join(DEFAULT_SOURCES)}).",
    )
    parser.add_argument("--raw-dir", type=Path, default=data.RAW_DIR, help="Output directory.")
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    args = parser.parse_args(argv)

    unknown = set(args.sources) - {*SOURCES, "all"}
    if unknown:
        parser.error(f"unknown source(s): {', '.join(sorted(unknown))}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    selected = set(args.sources or DEFAULT_SOURCES)
    if "all" in selected:
        selected = set(SOURCES)

    failures = 0
    for source in SOURCES:
        if source not in selected:
            continue
        try:
            if source == "lending-club":
                data.download_lending_club(args.raw_dir, force=args.force)
            elif source == "fred":
                data.download_fred(raw_dir=args.raw_dir, force=args.force)
            else:
                data.download_german_credit(args.raw_dir, force=args.force)
        except data.DataDownloadError as exc:
            logging.error("%s: %s", source, exc)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
