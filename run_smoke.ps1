# Local smoke test (Windows): 2 subjects, 1 run each (~0.7 GB download, BOLD deleted after use).
# LOSO with 2 subjects only tests the plumbing; the scores are meaningless.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
$subs = @("001", "002")

& $py -m pytest -q;                                                         if (-not $?) { exit 1 }
& $py scripts/01_download.py    --subjects $subs --limit-runs 1;            if (-not $?) { exit 1 }
& $py scripts/02_eeg_features.py --subjects $subs --limit-runs 1;           if (-not $?) { exit 1 }
& $py scripts/03_fmri_roi.py    --subjects $subs --limit-runs 1;            if (-not $?) { exit 1 }
& $py scripts/04_build_table.py --subjects $subs --limit-runs 1;            if (-not $?) { exit 1 }
& $py scripts/05_model.py       --subjects $subs --label both --n-null 20;  if (-not $?) { exit 1 }
Write-Host "Smoke run done. Open outputs/qc/index.html to check the masks."
