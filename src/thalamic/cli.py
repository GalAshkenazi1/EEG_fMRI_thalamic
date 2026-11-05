"""Shared command-line arguments for the pipeline scripts."""

from __future__ import annotations

import argparse
import logging

from . import config


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-dir", default=None,
                   help="Data root (default: $THAL_DATA_DIR or <repo>/data)")
    p.add_argument("--output-dir", default=None,
                   help="Outputs root (default: $THAL_OUTPUT_DIR or <repo>/outputs)")
    p.add_argument("--subjects", nargs="+", default=["all"],
                   help="Subject labels, e.g. 001 002, or 'all'")
    p.add_argument("--limit-runs", type=int, default=None,
                   help="Use at most N runs per subject (smoke tests)")
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def resolve_subjects(subjects: list[str]) -> list[str]:
    if subjects == ["all"] or subjects == "all":
        return list(config.ALL_SUBJECTS)
    return [f"{int(s):03d}" for s in subjects]


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("botocore", "s3fs", "aiobotocore", "fsspec", "urllib3", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
