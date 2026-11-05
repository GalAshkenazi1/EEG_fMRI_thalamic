"""Step 2: preprocess cleaned EEG, cut TR-locked windows, extract eegdash features.

Writes one parquet per run to <data>/derived/eeg_features/<stem>.parquet
(one row per TR: entity keys, tr index, CWL RMS, feature columns) and an EEG
PSD QC figure per run.
"""

import logging

from joblib import Parallel, delayed

from thalamic import cli, config, eeg, features, records

log = logging.getLogger("02_eeg_features")
KEYS = ("subject", "session", "task", "run", "stem")


def process(row, paths, overwrite=False):
    cli.setup_logging()
    out = paths.eeg_features / f"{row.stem}.parquet"
    psd_png = (paths.qc / f"sub-{row.subject}" / f"ses-{row.session}" /
               f"{row.stem}_desc-eegpsd.png")
    if out.exists() and psd_png.exists() and not overwrite:
        return out
    set_path = paths.raw / row.eeg_set
    if not set_path.exists():
        log.warning("%s: cleaned EEG missing, skipped", row.stem)
        return out if out.exists() else None
    raw0 = eeg.load_clean_eeg(set_path)
    if out.exists() and not overwrite:  # features cached, only the QC image is missing
        eeg.plot_psd_qc(eeg.preprocess_raw(raw0), psd_png,
                        title=f"{row.stem}: EEG PSD (31-ch mean)")
        return out
    win = int(round(config.TR * raw0.info["sfreq"]))
    starts = eeg.tr_marker_samples(raw0)
    starts = starts[starts + win <= raw0.n_times]
    raw = eeg.preprocess_raw(raw0)
    desc = {k: row[k] for k in KEYS}
    windows = eeg.make_tr_windows(raw, starts, desc)
    df = features.extract(windows, n_jobs=1)
    df["tr"] = df["i_trial_in_dataset"].astype(int)
    df["cwl_rms"] = eeg.cwl_rms_per_tr(raw0, starts, win)[df["tr"].to_numpy()]
    for k, v in desc.items():
        df[k] = v
    feat_cols = features.feature_columns(df)
    df[list(KEYS) + ["tr", "cwl_rms"] + feat_cols].to_parquet(out, index=False)
    eeg.plot_psd_qc(raw, psd_png, title=f"{row.stem}: EEG PSD (31-ch mean)")
    log.info("%s: %d windows x %d features", row.stem, len(df), len(feat_cols))
    return out


def main():
    p = cli.base_parser(__doc__)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    cli.setup_logging(a.verbose)
    paths = config.get_paths(a.data_dir, a.output_dir)
    runs = records.load_runs(paths, cli.resolve_subjects(a.subjects), a.limit_runs)
    outs = Parallel(n_jobs=a.n_jobs)(delayed(process)(r, paths, a.overwrite)
                                     for _, r in runs.iterrows())
    log.info("EEG features ready for %d/%d runs", sum(o is not None for o in outs), len(runs))


if __name__ == "__main__":
    main()
