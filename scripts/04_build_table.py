"""Step 4: align EEG features with ROI time series -> modeling table (both label types)."""

import pandas as pd

from thalamic import align, cli, config, features, records


def main():
    a = cli.base_parser(__doc__).parse_args()
    cli.setup_logging(a.verbose)
    paths = config.get_paths(a.data_dir, a.output_dir)
    runs = records.load_runs(paths, cli.resolve_subjects(a.subjects), a.limit_runs)
    eeg_files = [f for f in (paths.eeg_features / f"{s}.parquet" for s in runs.stem) if f.exists()]
    roi_files = [f for f in (paths.roi / f"{s}.parquet" for s in runs.stem) if f.exists()]
    eeg_df = pd.concat([pd.read_parquet(f) for f in eeg_files], ignore_index=True)
    roi_df = pd.concat([pd.read_parquet(f) for f in roi_files], ignore_index=True)
    feat_cols = features.feature_columns(eeg_df)
    table = align.build_table(eeg_df, roi_df, feat_cols)
    table.to_parquet(paths.table, index=False)
    print(f"{len(table)} windows x {len(feat_cols)} features -> {paths.table}")
    print(table.groupby("subject").agg(windows=("tr", "size"), runs=("stem", "nunique"),
                                       thal_active_rate=("thal_active", "mean"))
          .round(3).to_string())


if __name__ == "__main__":
    main()
