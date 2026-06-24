import io
import json
import os
import sys
import tarfile

from scipy.io import mmread
from scipy.sparse import issparse


def inspect_archive(path):
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".mtx"):
                fh = tar.extractfile(member)
                if fh is None:
                    break
                header = fh.readline().decode("utf-8").strip().lower()
                fh.seek(0)
                matrix = mmread(io.BytesIO(fh.read()))
                nrows, ncols = matrix.shape
                nnz = int(matrix.nnz if issparse(matrix) else (matrix != 0).sum())
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


def main():
    root = "/mnt/h/suitesparse"
    records = []
    for filename in sorted(os.listdir(root)):
        if not filename.endswith(".tar.gz"):
            continue
        record = inspect_archive(os.path.join(root, filename))
        if record is not None:
            records.append(record)
    json.dump(records, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
