#!/usr/bin/env python
"""Stream the full HF CARLA multimodal dataset in small batches (low RAM), count rows
per ``map_name``, and list **which global rows** belong to Town01 (and optional towns).

Dataset: https://huggingface.co/datasets/immanuelpeter/carla-autopilot-multimodal-dataset

* Reads only lightweight columns: ``map_name``, ``run_id``, ``frame`` (no images/LiDAR).
* ``global_row_index`` is 0-based, monotonic across **all processed splits** in order
  (train, validation, test by default).

Auth — set env before running (never commit tokens)::

    PowerShell:  $env:HF_TOKEN = "hf_..."
    See also:    huggingface-cli login

Usage::

    pip install datasets
    cd .../PythonAPI/examples

    # Full ~82k rows (train+val+test), chunked internally by --batch-size (default 128).
    python hf_probe_carla_maps.py --output-dir hf_probe_out

    # Quick test only:
    python hf_probe_carla_maps.py --max-rows-per-split 500

``--batch-size 128`` = read **128 rows per network decode chunk** (saves RAM). It is
**not** a cap on total rows — the run continues until every split is exhausted.

Press Ctrl+C to stop early (partial summary is still written).
"""
from __future__ import print_function

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from itertools import islice

try:
    from datasets import load_dataset
except ImportError:
    print("Install: pip install datasets", file=sys.stderr)
    sys.exit(1)

from hf_token_env import get_hf_hub_token

DEFAULT_SPLITS = ("train", "validation", "test")
# Substrings to classify rows into Town01..Town10 buckets (CARLA map paths)
TOWN_TAGS = tuple("Town{:02d}".format(i) for i in range(1, 11))


def main():
    p = argparse.ArgumentParser(
        description="Stream HF CARLA dataset: map stats + Town row indices (batched)")
    p.add_argument(
        "--dataset",
        default="immanuelpeter/carla-autopilot-multimodal-dataset")
    p.add_argument(
        "--splits", nargs="+", default=list(DEFAULT_SPLITS),
        help="HF splits to scan in order")
    p.add_argument(
        "--batch-size", type=int, default=128,
        help="rows per inner read chunk (memory vs network); NOT a total row limit")
    p.add_argument(
        "--log-every", type=int, default=5000,
        help="print progress every N rows per split (0 = log every 128-row chunk)")
    p.add_argument(
        "--max-rows-per-split", type=int, default=None,
        help="optional cap per split for debugging (default: None = entire split, ~67k+8.4k+7.2k)")
    p.add_argument(
        "--output-dir", default="hf_probe_out",
        help="summary JSON + town CSVs")
    p.add_argument(
        "--detail-towns", nargs="*", default=["Town01"],
        metavar="TownXX",
        help="write rows_<Tag>.csv for rows whose map_name contains that substring")
    args = p.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    token = get_hf_hub_token()
    print("[probe] Dataset:", args.dataset, flush=True)
    print("[probe] HF auth:", "token from env" if token else "anonymous", flush=True)
    print("[probe] Splits:", list(args.splits), flush=True)
    print("[probe] Inner chunk size (batch-size):", args.batch_size, flush=True)
    print("[probe] Progress log every (rows/split):",
          args.log_every if args.log_every else "each chunk", flush=True)
    if args.max_rows_per_split is None:
        print("[probe] MODE: FULL STREAM — no row cap (expect ~82k rows all splits).", flush=True)
    else:
        print("[probe] MODE: capped at {} rows PER SPLIT (debug).".format(
            args.max_rows_per_split), flush=True)

    load_kw = {"streaming": True}
    if token:
        load_kw["token"] = token

    cols = ["map_name", "run_id", "frame"]

    counts_by_map = defaultdict(int)
    counts_by_town_tag = defaultdict(int)

    detail_towns = [t.strip() for t in args.detail_towns if t.strip()]
    detail_handles = {}
    for tag in detail_towns:
        safe = tag.replace(os.sep, "_").replace("/", "_")
        path = os.path.join(out_dir, "rows_{}.csv".format(safe))
        f = open(path, "w", newline="")
        w = csv.writer(f)
        w.writerow(["global_row_index", "split", "run_id", "frame", "map_name"])
        detail_handles[tag] = (f, w)
        print("[probe] Detail file ({}):".format(tag), path, flush=True)

    global_idx = 0
    total_rows = 0
    per_split_totals = {}

    try:
        for split_name in args.splits:
            print("[probe] === split: {} ===".format(split_name), flush=True)
            ds = load_dataset(args.dataset, split=split_name, **load_kw)
            try:
                ds = ds.select_columns(cols)
            except Exception as e:
                print("[probe] select_columns warning:", e, flush=True)

            it = iter(ds)
            bs = max(1, args.batch_size)
            cap = args.max_rows_per_split
            rows_this_split = 0
            last_log = 0
            first_chunk = True

            while True:
                if cap is not None and rows_this_split >= cap:
                    break
                need = bs
                if cap is not None:
                    rem = cap - rows_this_split
                    if rem <= 0:
                        break
                    need = min(need, rem)

                chunk = list(islice(it, need))
                if not chunk:
                    break

                for row in chunk:
                    mraw = row.get("map_name")
                    m = str(mraw) if mraw is not None else ""
                    rid = row.get("run_id", "")
                    if rid is None:
                        rid = ""
                    fr = row.get("frame", "")
                    if fr is None:
                        fr = ""

                    counts_by_map[m] += 1
                    for tg in TOWN_TAGS:
                        if tg in m:
                            counts_by_town_tag[tg] += 1
                            break

                    for tag in detail_towns:
                        if tag in m:
                            detail_handles[tag][1].writerow(
                                [global_idx, split_name, rid, fr, m])

                    global_idx += 1
                    total_rows += 1
                    rows_this_split += 1

                per_split_totals[split_name] = rows_this_split

                log_every = args.log_every
                if first_chunk:
                    first_chunk = False
                    print("[probe]   {:8s}  first chunk OK  (+{} rows)  — full split streams until EOF.".format(
                        split_name, len(chunk)), flush=True)

                should_log = False
                if log_every <= 0:
                    should_log = True
                elif rows_this_split - last_log >= log_every:
                    should_log = True
                    last_log = (rows_this_split // log_every) * log_every

                if should_log:
                    print("[probe]   {:8s}  this_split {:9d}  total_all_splits {:9d}".format(
                        split_name, rows_this_split, total_rows), flush=True)

            print("[probe]   {:8s}  SPLIT DONE  rows {:9d}  total_all_splits {:9d}".format(
                split_name, rows_this_split, total_rows), flush=True)

    except KeyboardInterrupt:
        print("[probe] KeyboardInterrupt — saving partial summary.", flush=True)

    for tag, (f, _) in detail_handles.items():
        f.close()

    summary = {
        "dataset": args.dataset,
        "splits_scanned": list(args.splits),
        "total_rows": total_rows,
        "rows_per_split": per_split_totals,
        "unique_map_names": len(counts_by_map),
        "counts_by_map_name": dict(sorted(counts_by_map.items(), key=lambda x: -x[1])),
        "counts_by_town_tag_substring": dict(sorted(counts_by_town_tag.items())),
        "town_detail_tags_written": detail_towns,
        "notes": (
            "global_row_index is 0..N-1 in stream order across splits. "
            "Town tag counts use first matching Town01..Town10 substring in map_name."
        ),
    }
    sum_path = os.path.join(out_dir, "map_summary.json")
    with open(sum_path, "w") as jf:
        json.dump(summary, jf, indent=2)

    txt_path = os.path.join(out_dir, "map_counts.txt")
    with open(txt_path, "w") as tf:
        tf.write("Total rows: {}\n\n".format(total_rows))
        tf.write("Per-split:\n")
        for k, v in per_split_totals.items():
            tf.write("  {} : {}\n".format(k, v))
        tf.write("\nTown01..Town10 (substring in map_name):\n")
        for tg in TOWN_TAGS:
            tf.write("  {} : {}\n".format(tg, counts_by_town_tag.get(tg, 0)))
        tf.write("\nFull map_name counts (sorted desc):\n")
        for name, cnt in sorted(counts_by_map.items(), key=lambda x: -x[1]):
            tf.write("  {:6d}  {}\n".format(cnt, name))

    print("[probe] Done. total_rows:", total_rows, flush=True)
    print("[probe] Wrote:", sum_path, flush=True)
    print("[probe] Wrote:", txt_path, flush=True)


if __name__ == "__main__":
    main()
