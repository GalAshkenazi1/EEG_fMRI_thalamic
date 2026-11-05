"""Step 1: discover runs with eegdash and download cleaned EEG + T1 (+ optionally BOLD).

By default the raw BOLD runs are NOT downloaded here: 03_fmri_roi.py fetches
each BOLD file just before processing it and deletes it afterwards, so disk use
stays bounded. Use --with-bold to prefetch everything, and --dry-run to see the
total size before downloading.
"""

import logging

from thalamic import cli, config, records

log = logging.getLogger("01_download")


def main():
    p = cli.base_parser(__doc__)
    p.add_argument("--with-bold", action="store_true", help="also download raw BOLD now")
    p.add_argument("--dry-run", action="store_true", help="only list files and sizes")
    a = p.parse_args()
    cli.setup_logging(a.verbose)
    paths = config.get_paths(a.data_dir, a.output_dir)
    runs = records.load_runs(paths, cli.resolve_subjects(a.subjects), a.limit_runs)
    log.info("%d runs from %d subjects", len(runs), runs.subject.nunique())

    files = records.files_for_runs(runs, include_bold=a.with_bold or a.dry_run)
    rep = records.download(files, paths, dry_run=a.dry_run)
    print(rep.groupby("status").size_MB.agg(["count", "sum"]).round(1).to_string())
    if a.dry_run:
        bold = rep[rep.file.str.endswith("_bold.nii.gz")].size_MB.sum()
        print(f"\nTotal planned: {rep.size_MB.sum() / 1e3:.1f} GB "
              f"(BOLD: {bold / 1e3:.1f} GB, streamed and deleted by step 03)")


if __name__ == "__main__":
    main()
