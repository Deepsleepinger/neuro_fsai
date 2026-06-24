import argparse
import json
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path


SCALES = [
    ("s1", 100, 499),
    ("s2", 500, 999),
    ("s3", 1000, 1999),
    ("s4", 2000, 4999),
    ("s5", 5000, 20000),
]


def family_of(name):
    if "_" in name:
        return name.split("_", 1)[0]
    return name


def bucket_of(nrows):
    for label, lo, hi in SCALES:
        if lo <= nrows <= hi:
            return label
    return None


def load_legacy_exclusions(path):
    with open(path) as f:
        data = json.load(f)
    return set(data["train"]["names"]) | set(data["test"]["names"])


def load_eval_exclusions(path):
    with open(path) as f:
        data = json.load(f)
    excluded = set()
    for rows in data["scales"].values():
        excluded.update(row["name"] for row in rows)
    return excluded


def inspect_archive_header(path):
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".mtx"):
                continue
            fh = tar.extractfile(member)
            if fh is None:
                return None

            header = fh.readline().decode("utf-8").strip().lower()
            line = fh.readline().decode("utf-8").strip()
            while line.startswith("%"):
                line = fh.readline().decode("utf-8").strip()

            parts = line.split()
            if len(parts) < 3:
                return None

            nrows = int(parts[0])
            ncols = int(parts[1])
            nnz = int(parts[2])
            return {
                "name": os.path.basename(path).replace(".tar.gz", ""),
                "path": path,
                "nrows": nrows,
                "ncols": ncols,
                "nnz": nnz,
                "is_square": nrows == ncols,
                "is_general": "general" in header,
                "is_real": "complex" not in header and "pattern" not in header,
                "family": family_of(os.path.basename(path).replace(".tar.gz", "")),
            }
    return None


def collect_candidates(root, excluded_names):
    by_scale = {label: [] for label, _, _ in SCALES}
    for filename in sorted(os.listdir(root)):
        if not filename.endswith(".tar.gz"):
            continue
        name = filename.replace(".tar.gz", "")
        if name in excluded_names:
            continue

        info = inspect_archive_header(os.path.join(root, filename))
        if info is None:
            continue
        if not info["is_square"] or not info["is_general"] or not info["is_real"]:
            continue

        scale = bucket_of(info["nrows"])
        if scale is None:
            continue

        info["scale"] = scale
        by_scale[scale].append(info)

    for scale, rows in by_scale.items():
        by_scale[scale] = sorted(rows, key=lambda row: (row["family"], row["nrows"], row["nnz"], row["name"]))
    return by_scale


def round_robin_family_select(rows, count):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)

    families = sorted(grouped)
    selected = []
    while len(selected) < count:
        progressed = False
        for family in families:
            if not grouped[family]:
                continue
            selected.append(grouped[family].pop(0))
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def copy_selected(rows, split, copy_root):
    split_root = copy_root / split
    split_root.mkdir(parents=True, exist_ok=True)
    copied = []
    for row in rows:
        dest = split_root / f"{row['name']}.tar.gz"
        if not dest.exists():
            shutil.copy2(row["path"], dest)
        enriched = dict(row)
        enriched["split"] = split
        enriched["copied_to"] = str(dest)
        copied.append(enriched)
    return copied


def counts_by_scale(rows):
    counts = {label: 0 for label, _, _ in SCALES}
    for row in rows:
        counts[row["scale"]] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=str, default="/mnt/h/suitesparse")
    parser.add_argument("--legacy-used-json", type=str, default="results/used_matrices.json")
    parser.add_argument("--eval-json", type=str, default="results/selected_eval_matrices.json")
    parser.add_argument("--train-per-scale", type=int, default=16)
    parser.add_argument("--val-per-scale", type=int, default=4)
    parser.add_argument(
        "--copy-root",
        type=str,
        default="prepared/suitesparse_balanced_v1/archives",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="results/suitesparse_balanced_trainval_manifest.json",
    )
    args = parser.parse_args()

    legacy_excluded = load_legacy_exclusions(args.legacy_used_json)
    eval_excluded = load_eval_exclusions(args.eval_json)
    excluded_names = legacy_excluded | eval_excluded

    candidates = collect_candidates(args.source_root, excluded_names)
    train_rows = []
    val_rows = []

    for scale, _, _ in SCALES:
        target = args.train_per_scale + args.val_per_scale
        chosen = round_robin_family_select(candidates[scale], target)
        if len(chosen) < target:
            raise RuntimeError(
                f"Scale {scale} has only {len(chosen)} eligible matrices, "
                f"but {target} were requested."
            )
        val_rows.extend(chosen[:args.val_per_scale])
        train_rows.extend(chosen[args.val_per_scale:])

    copy_root = Path(args.copy_root)
    copied_train = copy_selected(train_rows, "train", copy_root)
    copied_val = copy_selected(val_rows, "val", copy_root)

    manifest = {
        "source_root": args.source_root,
        "copy_root": str(copy_root),
        "excluded": {
            "legacy_count": len(legacy_excluded),
            "eval_count": len(eval_excluded),
            "total_unique_count": len(excluded_names),
        },
        "config": {
            "train_per_scale": args.train_per_scale,
            "val_per_scale": args.val_per_scale,
            "scales": [
                {"name": label, "min_n": lo, "max_n": hi}
                for label, lo, hi in SCALES
            ],
        },
        "available_candidates": {
            scale: {
                "count": len(rows),
                "family_count": len({row["family"] for row in rows}),
            }
            for scale, rows in candidates.items()
        },
        "splits": {
            "train": copied_train,
            "val": copied_val,
        },
        "counts": {
            "train": counts_by_scale(copied_train),
            "val": counts_by_scale(copied_val),
        },
        "totals": {
            "train": len(copied_train),
            "val": len(copied_val),
        },
    }

    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["counts"], indent=2))
    print(json.dumps(manifest["totals"], indent=2))
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
