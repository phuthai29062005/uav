import argparse
import csv
import json
import sys
from pathlib import Path

COLUMNS = [
    "timestamp", "algo", "problem", "seed", "runtime_sec",
    "fes", "migd", "hv", "feasible_ratio", "status",
    "git_commit", "hostname", "params", "error_msg",
]


def convert(input_path, output_stream):
    writer = csv.DictWriter(output_stream, fieldnames=COLUMNS,
                            extrasaction="ignore")
    writer.writeheader()

    n_ok, n_bad = 0, 0
    with open(input_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"warning: skipped malformed line {lineno}: {e}",
                      file=sys.stderr)
                n_bad += 1
                continue

            if isinstance(entry.get("params"), dict):
                entry["params"] = json.dumps(entry["params"],
                                             ensure_ascii=False)

            unknown = set(entry) - set(COLUMNS)
            if unknown:
                print(f"warning: line {lineno} has unknown fields "
                      f"{unknown}, ignored", file=sys.stderr)

            writer.writerow(entry)
            n_ok += 1

    return n_ok, n_bad


def main():
    ap = argparse.ArgumentParser(description="Convert JSONL log to CSV")
    ap.add_argument("input", help="input .jsonl file")
    ap.add_argument("-o", "--output", default=None,
                    help="output .csv file (default: input with .csv "
                         "extension; '-' for stdout)")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input file not found: {input_path}",
              file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        output_path = input_path.with_suffix(".csv")
        out_stream = open(output_path, "w", encoding="utf-8", newline="")
        close_out = True
    elif args.output == "-":
        out_stream = sys.stdout
        close_out = False
    else:
        out_stream = open(args.output, "w", encoding="utf-8", newline="")
        close_out = True

    try:
        n_ok, n_bad = convert(input_path, out_stream)
    finally:
        if close_out:
            out_stream.close()

    print(f"converted {n_ok} rows, skipped {n_bad} malformed",
          file=sys.stderr)


if __name__ == "__main__":
    main()
