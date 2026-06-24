import json
import os
import sys

import numpy as np


def collect_names(split_root):
    names = set()
    for filename in sorted(os.listdir(split_root)):
        if not filename.endswith(".npy"):
            continue
        arr = np.load(os.path.join(split_root, filename), allow_pickle=True)
        for item in arr:
            meta = item.get("meta", {})
            if "name" in meta:
                names.add(meta["name"])
    return sorted(names)


def main():
    root = "/mnt/h/neuro_ilu/data"
    report = {}
    for split in ["train", "test"]:
        names = collect_names(os.path.join(root, split))
        report[split] = {
            "count": len(names),
            "names": names,
        }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
