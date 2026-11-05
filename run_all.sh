#!/usr/bin/env bash
# Full pipeline on all 24 subjects (intended for the Linux workstation).
#
#   THAL_DATA_DIR=/big/disk/thal N_JOBS=16 ./run_all.sh
#
# Disk: the cleaned EEG + T1s are ~5 GB. Raw BOLD (~65 GB in total) is streamed:
# each run is downloaded, processed and deleted (pass KEEP_BOLD=1 to keep it).
set -euo pipefail
cd "$(dirname "$0")"

N_JOBS="${N_JOBS:-$(nproc)}"
N_NULL="${N_NULL:-200}"
KEEP=""; [[ "${KEEP_BOLD:-0}" == "1" ]] && KEEP="--keep-bold"
# ANTs is memory-hungry (~3 GB per run in flight): cap parallel fMRI subjects.
FMRI_JOBS="${FMRI_JOBS:-$(( N_JOBS < 6 ? N_JOBS : 6 ))}"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
  .venv/bin/pip install -q -e .
fi
PY=.venv/bin/python

$PY -m pytest -q
$PY scripts/01_download.py   --subjects all --dry-run
$PY scripts/01_download.py   --subjects all
$PY scripts/02_eeg_features.py --subjects all --n-jobs "$N_JOBS"
$PY scripts/03_fmri_roi.py   --subjects all --n-jobs "$FMRI_JOBS" $KEEP
$PY scripts/04_build_table.py --subjects all
$PY scripts/05_model.py      --subjects all --label both --n-null "$N_NULL" --n-jobs "$N_JOBS"
$PY -m jupyter nbconvert --to notebook --execute --inplace notebooks/thalamus_from_eeg.ipynb

echo "Done. QC: outputs/qc/index.html | figures: outputs/figures | results: outputs/results"
