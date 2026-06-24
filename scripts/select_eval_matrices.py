import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path


SCALES = [
    ("s1", 100, 499),
    ("s2", 500, 999),
    ("s3", 1000, 1999),
    ("s4", 2000, 4999),
    ("s5", 5000, 20000),
]


def load_used_names(path, split):
    with open(path) as f:
        data = json.load(f)
    if split == "train":
        return set(data["train"]["names"])
    if split == "test":
        return set(data["test"]["names"])
    if split == "all":
        return set(data["train"]["names"]) | set(data["test"]["names"])
    raise ValueError(f"unknown exclusion split: {split}")


def inspect_archive(path):
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".mtx"):
                fh = tar.extractfile(member)
                if fh is None:
                    break
                header = fh.readline().decode("utf-8").strip().lower()
                line = fh.readline().decode("utf-8").strip()
                while line.startswith("%"):
                    line = fh.readline().decode("utf-8").strip()
                parts = line.split()
                if len(parts) < 3:
                    return None
                nrows, ncols, nnz = map(int, parts[:3])
                return {
                    "name": os.path.basename(path).replace(".tar.gz", ""),
                    "path": path,
                    "nrows": int(nrows),
                    "ncols": int(ncols),
                    "nnz": nnz,
                    "is_square": bool(nrows == ncols),
                    "is_general": "general" in header,
                }
    return None


def bucket_of(n):
    for label, lo, hi in SCALES:
        if lo <= n <= hi:
            return label
    return None


def family_of(name):
    if "_" in name:
        return name.split("_", 1)[0]
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="/mnt/h/suitesparse")
    parser.add_argument("--used-json", default="results/used_matrices.json")
    parser.add_argument("--exclude-split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--out-root", default="eval_matrices")
    parser.add_argument("--output-json", default="results/selected_eval_matrices.json")
    parser.add_argument("--per-scale", type=int, default=4)
    args = parser.parse_args()

    root = args.source_root
    used = load_used_names(args.used_json, args.exclude_split)
    out_root = Path(args.out_root)
    out_root.mkdir(exist_ok=True)

    selected = {label: [] for label, _, _ in SCALES}
    selected_families = {label: set() for label, _, _ in SCALES}
    for filename in sorted(os.listdir(root)):
        if not filename.endswith(".tar.gz"):
            continue
        path = os.path.join(root, filename)
        info = inspect_archive(path)
        if info is None:
            continue
        if not info["is_square"] or not info["is_general"]:
            continue
        if info["name"] in used:
            continue

        bucket = bucket_of(info["nrows"])
        if bucket is None or len(selected[bucket]) >= args.per_scale:
            continue
        family = family_of(info["name"])
        if family in selected_families[bucket]:
            continue

        dest = out_root / f"{info['name']}.tar.gz"
        if not dest.exists():
            shutil.copy2(path, dest)
        info["copied_to"] = str(dest)
        selected[bucket].append(info)
        selected_families[bucket].add(family)

        if all(len(selected[label]) >= args.per_scale for label, _, _ in SCALES):
            break

    report = {
        "source_root": root,
        "excluded_split": args.exclude_split,
        "excluded_count": len(used),
        "per_scale": args.per_scale,
        "scales": {label: rows for label, rows in selected.items()},
        "total_selected": sum(len(rows) for rows in selected.values()),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
