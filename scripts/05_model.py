"""Step 5: LOSO models, circular-shift null, controls/ablations, figures."""

import logging

import pandas as pd

from thalamic import cli, config, features, model, plots

log = logging.getLogger("05_model")


def main():
    p = cli.base_parser(__doc__)
    p.add_argument("--label", choices=["float", "binary", "both"], default="both",
                   help="window labels: float z-score, binary (>P75 active), or both")
    p.add_argument("--n-null", type=int, default=config.N_NULL,
                   help="circular-shift permutations (0 to skip)")
    a = p.parse_args()
    cli.setup_logging(a.verbose)
    paths = config.get_paths(a.data_dir, a.output_dir)
    table = pd.read_parquet(paths.table)
    table = table[table.subject.isin(cli.resolve_subjects(a.subjects))].reset_index(drop=True)
    feat_cols = features.feature_columns(table)
    modes = ["float", "binary"] if a.label == "both" else [a.label]
    results = {m: model.run_mode(table, feat_cols, m, a.n_null, a.n_jobs) for m in modes}
    model.save_results(results, paths.results)
    figs = plots.make_all(results, paths.figures)
    for m, r in results.items():
        print(f"\n=== {m} labels ===")
        print(pd.DataFrame(r["summary"]).T.round(4).to_string())
    log.info("results -> %s ; %d figures -> %s", paths.results, len(figs), paths.figures)


if __name__ == "__main__":
    main()
