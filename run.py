#!/usr/bin/env python
# multi_run_icssl.py
"""
Run exp_icssl_grid.py's inner loop N times (different seeds) and
report mean + variance of accuracies for each unlabelled‑block size U.
"""

import numpy as np, pandas as pd, argparse, random, time
from rich import print
from usb_icssl_eval import run_icssl_once

# ------------- GRID DEFINITION ---------------------------------
MODELS = {
    "Qwen‑3‑8B": dict(
        name="Qwen/Qwen3-8B",   # change to your repo id
        quant4=True,
        dtype="bfloat16",
    ),
    # add other models if you like
}

UNLAB_COUNTS = [32, 64, 128, 256, 512]
# ---------------------------------------------------------------


def main(runs: int = 5, task: str = "ag_news", k: int = 4):
    rows = []
    t0 = time.time()

    for r in range(runs):
        seed = 42 + r        # different seed each repeat
        random.seed(seed)
        np.random.seed(seed)

        print(f"[bold yellow]=== Run {r+1}/{runs} (seed={seed}) ===[/]")
        for label, cfg in MODELS.items():
            for u in UNLAB_COUNTS:
                acc = run_icssl_once(
                    cfg["name"],
                    task=task,
                    k=k,
                    u=u,
                    dtype=cfg["dtype"],
                    quant4=cfg["quant4"],
                    seed=seed,
                    verbose=False,      # flip to True for debugging
                )
                rows.append(
                    dict(run=r, model=label, unlab=u, acc=acc, seed=seed)
                )
                print(f"{label:10} U={u:<4} acc={acc:.4f}")

    # --------- save raw results ---------------------------------
    df = pd.DataFrame(rows)
    df.to_csv("all_runs.csv", index=False)

    # --------- compute mean & variance --------------------------
    stats = (
        df.groupby(["model", "unlab"])["acc"]
        .agg(["mean", "var"])
        .reset_index()
        .rename(columns={"var": "variance"})
    )
    stats.to_csv("stats.csv", index=False)

    # --------- pretty summary -----------------------------------
    print("\n[bold green]=== Mean ± Variance after "
          f"{runs} runs ({time.time() - t0:.1f}s) ===[/]")
    for label, sub in stats.groupby("model"):
        print(f"[bold]{label}[/]")
        for _, row in sub.iterrows():
            print(f"  U={int(row.unlab):<4} "
                  f"mean={row.mean:.4f}  var={row.variance:.6f}")
        print()

    print("Saved raw data → all_runs.csv")
    print("Saved stats    → stats.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5, help="# independent repeats")
    ap.add_argument("--task", default="ag_news")
    ap.add_argument("--k", type=int, default=4, help="# labelled demos/class")
    args = ap.parse_args()
    main(runs=args.runs, task=args.task, k=args.k)
