#!/usr/bin/env python
# exp_icssl_grid.py
import pandas as pd, matplotlib.pyplot as plt
from rich import print
from usb_icssl_eval import run_icssl_once


MODELS = {
    # "Llama‑3‑8B": dict(
    #     name="meta-llama/Meta-Llama-3-8B-Instruct",
    #     quant4=True,           # 4‑bit saves ~50 % VRAM; bf16 also OK on H100
    #     dtype="bfloat16",
    # ),

    "Qwen‑3‑8B": dict(
        name="Qwen/Qwen3-8B",
        quant4=False,          # fits bf16
        dtype="bfloat16",
    ),
}

# UNLAB_COUNTS = [32, 64, 128, 256, 512]
UNLAB_COUNTS = [4, 16, 64, 256]

rows = []
for lbl, cfg in MODELS.items():
    for u in UNLAB_COUNTS:
        acc = run_icssl_once(cfg["name"],
                             task="ag_news",
                             k=4, u=u,
                             dtype=cfg["dtype"],
                             quant4=cfg["quant4"],
                             verbose=True) 
        rows.append(dict(model=lbl, unlab=u, acc=acc))
        print(f"[bold cyan]{lbl}[/]  U={u:<3}  →  acc={acc:.4f}")

df = pd.DataFrame(rows)
df.to_csv("usb_grid_results.csv", index=False)

# ---------- quick plot ----------
plt.figure()
for lbl, sub in df.groupby("model"):
    plt.plot(sub["unlab"], sub["acc"], marker="o", label=lbl)
plt.xlabel("# unlabeled queries in prompt")
plt.ylabel("accuracy on the unlabeled block")
plt.xscale("log", base=2)
plt.grid(True, which="both", ls=":")
plt.title("USB (AG News, K=4) — IC‑SSL accuracy vs. context size")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_vs_unlab.png", dpi=150)
print("\nSaved: usb_grid_results.csv  ·  accuracy_vs_unlab.png")
