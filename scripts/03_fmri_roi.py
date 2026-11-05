"""Step 3: BOLD -> motion correction, registration, masks, cleaned ROI time series.

Each BOLD run is downloaded just in time (if absent), processed, and deleted
afterwards unless --keep-bold is given. QC images go to <outputs>/qc and an
index page is written to <outputs>/qc/index.html.
"""

import logging
import os

from joblib import Parallel, delayed

from thalamic import cli, config, fmri, qc, records

log = logging.getLogger("03_fmri_roi")


def run_subject(subject, runs, paths, keep_bold):
    cli.setup_logging()
    sub_runs = runs[runs.subject == subject]
    # Runs without cleaned EEG (e.g. sub-003 ses-001 GradCPT, sub-023 ses-002) can never
    # enter the model table, so their ~400 MB BOLD is not downloaded or processed.
    has_eeg = [(paths.eeg_features / f"{s}.parquet").exists() for s in sub_runs.stem]
    for s in sub_runs.stem[[not h for h in has_eeg]]:
        log.warning("%s: no EEG features (run step 2 first / no cleaned EEG) - BOLD skipped", s)
    sub_runs = sub_runs[has_eeg]
    todo = sub_runs[[not (paths.roi / f"{s}.parquet").exists() for s in sub_runs.stem]]
    if todo.empty:
        return
    records.download([sub_runs.iloc[0].t1], paths)
    for _, r in todo.iterrows():
        records.download([r.bold], paths)  # just in time; deleted after processing
        fmri.process_subject(subject, todo[todo.stem == r.stem], paths, keep_bold)


def main():
    p = cli.base_parser(__doc__)
    p.add_argument("--keep-bold", action="store_true", help="do not delete raw BOLD after use")
    a = p.parse_args()
    cli.setup_logging(a.verbose)
    paths = config.get_paths(a.data_dir, a.output_dir)
    runs = records.load_runs(paths, cli.resolve_subjects(a.subjects), a.limit_runs)
    # share cores between parallel subjects and ANTs' internal threads
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
                          str(max(1, (os.cpu_count() or 1) // max(1, a.n_jobs))))
    fmri.mni_assets(paths.anat)  # build once before parallel workers start
    Parallel(n_jobs=a.n_jobs)(delayed(run_subject)(s, runs, paths, a.keep_bold)
                              for s in sorted(runs.subject.unique()))
    log.info("QC index: %s", qc.write_index(paths.qc, paths.roi / "qc_summary"))


if __name__ == "__main__":
    main()
