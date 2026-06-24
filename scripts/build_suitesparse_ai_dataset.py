"""Build a standardized SuiteSparse dataset for AI preconditioner training.

The script only reads archives from the source directory. Selected archives are
copied to a local prepared dataset directory, and a manifest records the exact
filters and split assignment.
"""

import argparse
import csv
import json
import os
import re
import shutil
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_SCALES = [
    ("s1", 1000, 1999),
    ("s2", 2000, 4999),
    ("s3", 5000, 9999),
    ("s4", 10000, 19999),
    ("s5", 20000, 30000),
]


def parse_scales(spec):
    if not spec:
        return DEFAULT_SCALES
    scales = []
    for item in spec.split(","):
        label, bounds = item.split(":", 1)
        lo, hi = bounds.split("-", 1)
        scales.append((label, int(lo), int(hi)))
    return scales


def group_and_matrix(name):
    if "_" not in name:
        return name, name
    group, matrix = name.split("_", 1)
    return group, matrix


def family_of(name):
    """Approximate SuiteSparse series key to avoid train/test leakage.

    SuiteSparse has many numbered series such as FIDAP_ex1/FIDAP_ex2 or
    DRIVCAV_cavity03/cavity04. The exact family metadata is not present inside
    local tarballs, so use a conservative filename-derived key.
    """
    group, matrix = group_and_matrix(name)
    key = matrix.lower()
    key = re.sub(r"([_-]?)(ex|problem|case|cavity|goodwin|rajat|g7jac|fs_?)\d+[a-z]?$", r"\1\2", key)
    key = re.sub(r"([_-]?)(\d+)[a-z]?$", "", key)
    key = re.sub(r"([_-]?)[a-z]\d+$", "", key)
    key = key.strip("_-")
    if not key:
        key = matrix.lower()
    return f"{group}_{key}"


def bucket_of(nrows, scales):
    for label, lo, hi in scales:
        if lo <= nrows <= hi:
            return label
    return None


def inspect_archive_header(path):
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".mtx"):
                continue
            fh = tar.extractfile(member)
            if fh is None:
                return None

            header = fh.readline().decode("utf-8", errors="replace").strip().lower()
            line = fh.readline().decode("utf-8", errors="replace").strip()
            while line.startswith("%"):
                line = fh.readline().decode("utf-8", errors="replace").strip()

            parts = line.split()
            if len(parts) < 3:
                return None

            nrows = int(parts[0])
            ncols = int(parts[1])
            nnz = int(parts[2])
            tokens = header.split()
            field = tokens[3] if len(tokens) > 3 else "unknown"
            symmetry = tokens[4] if len(tokens) > 4 else "unknown"
            name = os.path.basename(path).replace(".tar.gz", "")
            group, matrix = group_and_matrix(name)
            return {
                "name": name,
                "path": str(path),
                "member": member.name,
                "group": group,
                "matrix": matrix,
                "family": family_of(name),
                "nrows": nrows,
                "ncols": ncols,
                "nnz": nnz,
                "nnz_per_row": nnz / max(1, nrows),
                "field": field,
                "symmetry": symmetry,
                "is_square": nrows == ncols,
                "is_real_valued": field in {"real", "integer"},
                "is_pattern": field == "pattern",
                "is_complex": field == "complex",
                "is_general": symmetry == "general",
            }
    return None


def scan_archive_worker(path):
    try:
        return inspect_archive_header(path), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def collect_names_from_json(path):
    if not path or not Path(path).exists():
        return set()
    with open(path) as f:
        data = json.load(f)

    names = set()

    def visit(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get("name"), str):
                names.add(obj["name"])
            if isinstance(obj.get("names"), list):
                names.update(x for x in obj["names"] if isinstance(x, str))
            for value in obj.values():
                visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(data)
    return names


def load_metadata_catalog(path):
    if not path:
        return {}
    catalog = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name") or row.get("Name") or row.get("matrix") or row.get("Matrix")
            if name:
                catalog[name] = row
    return catalog


def load_scan_cache(path):
    if not path or not Path(path).exists() or Path(path).stat().st_size == 0:
        return None
    with open(path) as f:
        data = json.load(f)
    rows = data["records"] if isinstance(data, dict) and "records" in data else data
    return rows


def write_scan_cache(path, rows, source_root):
    if not path:
        return
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "source_root": source_root,
        "record_count": len(rows),
        "records": rows,
    }
    cache_path.write_text(json.dumps(cache, indent=2))


def scan_archives(args):
    cached = None if args.refresh_catalog else load_scan_cache(args.catalog_cache)
    if cached is not None:
        return cached, {"cache_hit": 1}

    archives = sorted(Path(args.source_root).glob("*.tar.gz"))
    rows = []
    rejected = defaultdict(int)
    if args.workers <= 1:
        for archive in archives:
            row, error = scan_archive_worker(archive)
            if error is not None:
                rejected["read_error"] += 1
            elif row is None:
                rejected["no_matrix"] += 1
            else:
                rows.append(row)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(scan_archive_worker, archive): archive for archive in archives}
            for future in as_completed(futures):
                row, error = future.result()
                if error is not None:
                    rejected["read_error"] += 1
                elif row is None:
                    rejected["no_matrix"] += 1
                else:
                    rows.append(row)

    rows = sorted(rows, key=lambda row: row["name"])
    write_scan_cache(args.catalog_cache, rows, args.source_root)
    return rows, dict(rejected)


def pass_filters(row, args, scales):
    if not row["is_square"]:
        return False, "not_square"
    if not row["is_real_valued"] or row["is_pattern"] or row["is_complex"]:
        return False, "not_real_valued"
    if args.require_general and not row["is_general"]:
        return False, "not_general"
    if row["nrows"] < args.min_n or row["nrows"] > args.max_n:
        return False, "size"
    if row["nnz_per_row"] > args.max_degree:
        return False, "degree"
    scale = bucket_of(row["nrows"], scales)
    if scale is None:
        return False, "scale"
    return True, scale


def scan_candidates(args, scales):
    excluded_names = set()
    for path in args.exclude_json:
        excluded_names.update(collect_names_from_json(path))

    metadata_catalog = load_metadata_catalog(args.catalog_csv)
    candidates = []
    rejected = defaultdict(int)
    scanned_rows, scan_rejected = scan_archives(args)
    rejected.update(scan_rejected)
    for row in scanned_rows:
        name = row["name"]
        if name in excluded_names:
            rejected["excluded_name"] += 1
            continue

        ok, reason_or_scale = pass_filters(row, args, scales)
        if not ok:
            rejected[reason_or_scale] += 1
            continue
        row["scale"] = reason_or_scale
        if row["name"] in metadata_catalog:
            row["catalog"] = metadata_catalog[row["name"]]
            row["kind"] = (
                metadata_catalog[row["name"]].get("kind")
                or metadata_catalog[row["name"]].get("Kind")
                or metadata_catalog[row["name"]].get("problem")
                or metadata_catalog[row["name"]].get("Problem")
                or ""
            )
        else:
            row["catalog"] = {}
            row["kind"] = ""
        candidates.append(row)

    return candidates, rejected, excluded_names


def split_targets(args, scales):
    return {
        "train": {label: args.train_per_scale for label, _, _ in scales},
        "val": {label: args.val_per_scale for label, _, _ in scales},
        "test": {label: args.test_per_scale for label, _, _ in scales},
    }


def stratum_key(row, stratify_by):
    if stratify_by == "kind":
        return row.get("kind") or row["group"]
    if stratify_by == "group":
        return row["group"]
    if stratify_by == "family":
        return row["family"]
    return "all"


def select_split_for_scale(rows, scale, targets, args, assigned_family):
    selected = {"train": [], "val": [], "test": []}
    rows = [row for row in rows if row["scale"] == scale]
    rows = sorted(rows, key=lambda r: (stratum_key(r, args.stratify_by), r["family"], r["nrows"], r["nnz"], r["name"]))

    split_order = ["test", "val", "train"]
    used_names = set()

    def fill_split(split, strict_family_limit):
        by_stratum = defaultdict(list)
        for row in rows:
            if row["name"] in used_names:
                continue
            assigned = assigned_family.get(row["family"])
            if assigned is not None and assigned != split:
                continue
            if strict_family_limit and any(x["family"] == row["family"] for x in selected[split]):
                continue
            by_stratum[stratum_key(row, args.stratify_by)].append(row)

        strata = sorted(by_stratum)
        while len(selected[split]) < targets[split][scale]:
            progressed = False
            for stratum in strata:
                bucket = by_stratum[stratum]
                while bucket and bucket[0]["name"] in used_names:
                    bucket.pop(0)
                if not bucket:
                    continue
                row = bucket.pop(0)
                if row["family"] not in assigned_family:
                    assigned_family[row["family"]] = split
                selected[split].append(row)
                used_names.add(row["name"])
                progressed = True
                if len(selected[split]) >= targets[split][scale]:
                    break
            if not progressed:
                break

    for split in split_order:
        fill_split(split, strict_family_limit=True)
        if len(selected[split]) < targets[split][scale]:
            fill_split(split, strict_family_limit=False)

    underfilled = {
        split: targets[split][scale] - len(selected[split])
        for split in split_order
        if len(selected[split]) < targets[split][scale]
    }
    return selected, underfilled


def copy_rows(rows, split, archives_root, dry_run):
    copied = []
    split_root = archives_root / split
    if not dry_run:
        split_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        enriched = dict(row)
        dest = split_root / f"{row['name']}.tar.gz"
        if not dry_run and not dest.exists():
            shutil.copy2(row["path"], dest)
        enriched["split"] = split
        enriched["copied_to"] = str(dest)
        copied.append(enriched)
    return copied


def counts_by(rows, key):
    out = defaultdict(int)
    for row in rows:
        out[row[key]] += 1
    return dict(sorted(out.items()))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split", "scale", "name", "group", "family", "nrows", "ncols",
        "nnz", "nnz_per_row", "field", "symmetry", "path", "copied_to",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="/mnt/h/suitesparse")
    parser.add_argument("--output-root", default="prepared/suitesparse_ai_v1")
    parser.add_argument("--manifest-path", default="results/suitesparse_ai_v1_manifest.json")
    parser.add_argument("--catalog-csv", default="")
    parser.add_argument("--catalog-cache", default="results/suitesparse_ai_catalog_cache.json")
    parser.add_argument("--refresh-catalog", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--exclude-json", action="append", default=[])
    parser.add_argument("--scales", default="")
    parser.add_argument("--min-n", type=int, default=1000)
    parser.add_argument("--max-n", type=int, default=30000)
    parser.add_argument("--max-degree", type=float, default=50.0)
    parser.add_argument("--require-general", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stratify-by", choices=["kind", "group", "family", "none"], default="group")
    parser.add_argument("--train-per-scale", type=int, default=100)
    parser.add_argument("--val-per-scale", type=int, default=10)
    parser.add_argument("--test-per-scale", type=int, default=10)
    parser.add_argument("--allow-underfill", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scales = parse_scales(args.scales)
    candidates, rejected, excluded_names = scan_candidates(args, scales)
    by_scale = defaultdict(list)
    for row in candidates:
        by_scale[row["scale"]].append(row)

    targets = split_targets(args, scales)
    assigned_family = {}
    splits = {"train": [], "val": [], "test": []}
    underfilled_all = {}
    for scale, _, _ in scales:
        selected, underfilled = select_split_for_scale(
            candidates, scale, targets, args, assigned_family)
        for split, rows in selected.items():
            splits[split].extend(rows)
        if underfilled:
            underfilled_all[scale] = underfilled

    if underfilled_all and not args.allow_underfill:
        raise RuntimeError(
            "Requested dataset is underfilled. Re-run with --allow-underfill "
            f"or lower per-scale targets. Underfilled: {underfilled_all}"
        )

    output_root = Path(args.output_root)
    archives_root = output_root / "archives"
    copied_splits = {
        split: copy_rows(rows, split, archives_root, args.dry_run)
        for split, rows in splits.items()
    }
    all_rows = copied_splits["train"] + copied_splits["val"] + copied_splits["test"]

    manifest = {
        "version": "suitesparse_ai_v1",
        "source_root": args.source_root,
        "output_root": str(output_root),
        "dry_run": args.dry_run,
        "filters": {
            "min_n": args.min_n,
            "max_n": args.max_n,
            "max_degree": args.max_degree,
            "require_general": args.require_general,
            "require_square": True,
            "require_real_valued": True,
        },
        "scales": [
            {"name": label, "min_n": lo, "max_n": hi}
            for label, lo, hi in scales
        ],
        "targets": targets,
        "stratify_by": args.stratify_by,
        "excluded_count": len(excluded_names),
        "scan": {
            "eligible_count": len(candidates),
            "eligible_by_scale": {label: len(by_scale[label]) for label, _, _ in scales},
            "rejected": dict(sorted(rejected.items())),
        },
        "underfilled": underfilled_all,
        "splits": copied_splits,
        "counts": {
            split: {
                "total": len(rows),
                "by_scale": counts_by(rows, "scale"),
                "by_group": counts_by(rows, "group"),
                "family_count": len({row["family"] for row in rows}),
            }
            for split, rows in copied_splits.items()
        },
    }

    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    write_csv(manifest_path.with_suffix(".csv"), all_rows)

    print(json.dumps({
        "eligible_by_scale": manifest["scan"]["eligible_by_scale"],
        "counts": manifest["counts"],
        "underfilled": underfilled_all,
        "manifest": str(manifest_path),
        "dry_run": args.dry_run,
    }, indent=2))


if __name__ == "__main__":
    main()
