#!/usr/bin/env python
# exp_icssl_grid.py
import pandas as pd, matplotlib.pyplot as plt
from rich import print
from usb_icssl_eval import run_icssl_once

MODELS = {
    "Mistral‑7B" : dict(name="mistralai/Mistral-7B-v0.2",
                        quant4=False, dtype="bfloat16"),
    "Llama‑2‑13B": dict(name="meta-llama/Llama-2-13b-chat-hf",
                        quant4=True,  dtype="bfloat16"),   # 4‑bit to fit comfortably
    "Qwen‑1.5‑7B": dict(name="Qwen/Qwen1.5-7B",
                        quant4=False, dtype="bfloat16"),
}

UNLAB_COUNTS = [32, 64, 128, 256, 512]

rows = []
for lbl, cfg in MODELS.items():
    for u in UNLAB_COUNTS:
        acc = run_icssl_once(cfg["name"],
                             task="ag_news",
                             k=4, u=u,
                             dtype=cfg["dtype"],
                             quant4=cfg["quant4"])
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
