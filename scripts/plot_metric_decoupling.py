import argparse
import csv
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PATTERN = re.compile(
    r"epoch=(?P<epoch>\d+).*?loss=(?P<loss>[-+0-9.eE]+).*?"
    r"(?:mse=(?P<mse>[-+0-9.eE]+).*?alg=(?P<alg>[-+0-9.eE]+).*?)?"
    r"solver_info=(?P<info>-?\d+).*?solver_iter=(?P<iter>\d+).*?"
    r"final_rel=(?P<rel>[-+0-9.eEnNaA]+)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot loss-vs-BiCGSTAB iteration decoupling from a Neuro-SPAI log.txt.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--title", default=None)
    parser.add_argument("--min-epoch", type=int, default=None)
    parser.add_argument("--max-epoch", type=int, default=None)
    return parser.parse_args()


def to_float(text):
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return float("nan")


def main():
    args = parse_args()
    log_path = pathlib.Path(args.log)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in log_path.read_text().splitlines():
        match = PATTERN.search(line)
        if not match:
            continue
        rows.append({
            "epoch": int(match.group("epoch")),
            "loss": to_float(match.group("loss")),
            "mse": to_float(match.group("mse")),
            "alg": to_float(match.group("alg")),
            "solver_info": int(match.group("info")),
            "solver_iter": int(match.group("iter")),
            "final_rel": to_float(match.group("rel")),
        })
    if args.min_epoch is not None:
        rows = [row for row in rows if row["epoch"] >= args.min_epoch]
    if args.max_epoch is not None:
        rows = [row for row in rows if row["epoch"] <= args.max_epoch]
    if not rows:
        raise SystemExit(f"No eval rows parsed from {log_path}")

    stem = log_path.parent.name
    if args.min_epoch is not None or args.max_epoch is not None:
        stem += f"_e{args.min_epoch or 'start'}-{args.max_epoch or 'end'}"
    csv_path = out_dir / f"{stem}_metric_decoupling.csv"
    png_path = out_dir / f"{stem}_metric_decoupling.png"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    epochs = [row["epoch"] for row in rows]
    loss = [row["mse"] if row["mse"] is not None else row["loss"] for row in rows]
    iters = [row["solver_iter"] for row in rows]
    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax1.plot(epochs, loss, color="#1f77b4", marker="o", linewidth=1.8, label="MSE/loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE / loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(epochs, iters, color="#d62728", marker="s", linewidth=1.8, label="BiCGSTAB iterations")
    ax2.set_ylabel("BiCGSTAB iterations", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    title = args.title or "Metric Decoupling: Loss vs Solver Iterations"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png_path)
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
