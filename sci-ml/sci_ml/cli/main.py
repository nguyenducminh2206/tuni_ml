import argparse
from importlib.metadata import version, PackageNotFoundError
from sci_ml.data.preview import preview_csv


def _pkg_version() -> str:
    for name in ("sci-ml", "sci_ml"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return "0.0.0"


def _load_cmd(args):
    preview_csv(args.path, label=args.label)  # nrows fixed at 5 by default


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sci-ml",
        description="sci-ml CLI (machine learning for science).",
    )
    parser.add_argument("-V", "--version", action="version",
                        version=f"sci-ml {_pkg_version()}")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # sci-ml load <path/to/file.csv> [--label y]
    p_load = sub.add_parser("load", help="preview a CSV file (no changes made)")
    p_load.add_argument("path", help="path to CSV file")
    p_load.add_argument("--label", help="label column name to count classes", default=None)
    p_load.set_defaults(func=_load_cmd)

    args = parser.parse_args()
    args.func(args)
