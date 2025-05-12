#!/usr/bin/env python
# usb_icssl_eval.py
"""
Evaluate one IC‑SSL prompt on a USB task and (optionally) print
the prompt + generation for debugging.
"""

from collections import defaultdict
import random, textwrap, torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)


LABEL_LINE = "TEXT: {text}\nLABEL: {label}\n"
UNLAB_LINE = "TEXT: {text}\nLABEL:"


# TEMPLATE = """
# You are an expert topic‑classifier. Think step‑by‑step **silently**.
# Never reveal chain‑of‑thought.

# Below are four possible categories:

# {label_desc}

# ────────────────  LABELED DEMOS  ────────────────
# {label_block}

# ────────────────  UNLABELED TEXTS  ──────────────
# {unlab_block}

# ────────────────  OUTPUT INSTRUCTIONS  ─────────
# First, reason *privately*.
# Then print exactly:
# ANSWER:
# <label‑1>
# <label‑2>
# …
# <label‑N>

# Where <label‑i> is one of: {label_list}.
# **Do not print anything else.**
# If you add explanations, extra blank lines, or markup
# (e.g. “<think>...</think>”), your answer will be graded zero.
# """.strip()

#################################################################
TEMPLATE = """
You are an expert text‑classifier.  Possible classes are:
{label_desc}

The following examples are ALREADY labelled.

{unlab_block}

---

First **think** silently.  **Do NOT reveal your reasoning.**
When you are certain, write:

ANSWER:
<category‑1>
<category‑2>
... (one label per line in the same order)

Do NOT output anything else.
""".strip()

#################################################################
# TEMPLATE = """
# You are an expert text‑classifier.  Possible classes are:
# {label_desc}

# The following examples are ALREADY labelled.

# {label_block}

# Now read ALL of the unlabelled texts below.
#  • First, think step‑by‑step, compare them with the patterns you saw 
#  in the labelled and unlabeled blocks, and decide the best category for each one. For each one, your thinking token should not exceed 100.
#  Remember that information in the unlabeled block can be utilized to improve your prediction.
#  • Write your reasoning INSIDE a <think> ... </think> block.
#  • AFTER the </think> tag, output ONLY the category names,
#    one per line, in the *same order* as the texts appear.

# {unlab_block}

# **<think>
# ... your analysis ...
# </think>
# <your labels here>
# **
# """.strip()




USB = {
    "ag_news": dict(
        hf="ag_news", txt="text", lab="label",
        labels=["World", "Sports", "Business", "Sci/Tech"]),
    "amazon_polarity": dict(
        hf="amazon_polarity", txt="content", lab="label",
        labels=["negative", "positive"]),
}

def build_prompt(l_demo, u_demo, labels):
    label_desc  = "\n".join(f"- {x}" for x in labels)
    label_block = "\n".join(
        LABEL_LINE.format(text=e["text"], label=labels[e["label"]])
        for e in l_demo)
    unlab_block = "\n".join(
        UNLAB_LINE.format(text=e["text"]) for e in u_demo)
    return TEMPLATE.format(**locals())

# ─────────────────────────  chat wrapper  ──────────────────────
def encode_prompt(tokenizer, plain_prompt):
    """Wrap plain prompt in chat template if the tokenizer supports it."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system",
             "content": "You are a helpful, expert text‑classifier."},
            {"role": "user", "content": plain_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt")
    return tokenizer(plain_prompt, return_tensors="pt")
# ───────────────────────────────────────────────────────────────

def run_icssl_once(model_name: str,
                   task: str  = "ag_news",
                   k: int     = 4,
                   u: int     = 128,
                   dtype: str = "bfloat16",
                   quant4: bool = False,
                   seed: int   = 42,
                   verbose: bool = False):
    """
    Run one IC‑SSL evaluation and return accuracy.
    If verbose=True, also print the prompt and raw generation.
    """
    meta = USB[task]
    rnd = random.Random(seed)

    # ---------- sample labelled + unlabelled --------------------
    train_ds = load_dataset(meta["hf"], split="train")
    per_class = defaultdict(list)
    for ex in train_ds:
        per_class[ex[meta["lab"]]].append(
            dict(text=ex[meta["txt"]].strip().replace("\n", " "),
                 label=ex[meta["lab"]]))
    labelled = sum((rnd.sample(pool, k) for pool in per_class.values()), [])

    test_ds = load_dataset(meta["hf"], split="test")
    unlab   = [dict(text=ex[meta["txt"]].strip().replace("\n", " "),
                    label=ex[meta["lab"]])
               for ex in rnd.sample(list(test_ds), u)]

    prompt = build_prompt(labelled, unlab, meta["labels"])

    # ---------- load model --------------------------------------
    if quant4:
        qcfg = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto",
            quantization_config=qcfg, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto",
            torch_dtype=getattr(torch, dtype),
            trust_remote_code=True)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True,
                                        trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # ---------- encode prompt (Chat vs raw) ---------------------
    inputs = encode_prompt(tok, prompt, model.device)

    # reduce VRAM on long contexts
    model.config.use_cache = False

    gen_ids = model.generate(
        **inputs,
        max_new_tokens= 100*u ,        # ≈ one token per label
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        pad_token_id=tok.eos_token_id,
    )

    #############################################
    # raw_out = tok.decode(gen_ids[0][inputs["input_ids"].shape[-1]:],
    #                  skip_special_tokens=True).strip()

    # after_think = raw_out.split("</think>")[-1]   # falls back to full text if no tag
    # preds = [ln.strip() for ln in after_think.splitlines() if ln.strip()][:u]
    #############################################
                     
    raw_out = tok.decode(gen_ids[0][inputs["input_ids"].shape[-1]:],
                         skip_special_tokens=True).strip()
    preds = [ln.strip() for ln in raw_out.splitlines() if ln.strip()][:u]

    lab2id = {l: i for i, l in enumerate(meta["labels"])}
    gold   = [ex["label"] for ex in unlab]
    guess  = [lab2id.get(p, -1) for p in preds]
    acc    = sum(int(p == g) for p, g in zip(guess, gold)) / len(gold)

    # ---------- diagnostics -------------------------------------
    if verbose:
        print("\n" + "="*28 + " PROMPT (truncated) " + "="*28)
        print(prompt[:800] + (" …" if len(prompt) > 800 else ""))
        print("="*75)
        print("RAW GENERATION:\n", raw_out)
        print("="*75)
        print("PARSED  vs  GOLD")
        for p, g in zip(preds[:10], gold[:10]):   # show first 10
            print(f"{p:12} | {meta['labels'][g]}")
        print(f"→ accuracy {acc:.4f}\n")

    torch.cuda.empty_cache()
    return acc

def encode_prompt(tokenizer, plain_prompt, device):
    """
    Return a dict that has at least 'input_ids' (and attention_mask) so
    we can pass it to model.generate(**inputs, …) without errors.
    """
    # Case A ▸ the tokenizer has a chat template (Qwen‑3, Qwen‑2, Llama‑3…)
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system",
             "content": "You are a helpful, expert text‑classifier."},
            {"role": "user", "content": plain_prompt},
        ]
        tpl_out = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Some tokenizers return a tensor, others a dict → normalise
        if isinstance(tpl_out, torch.Tensor):
            input_ids = tpl_out.to(device)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        else:
            return {k: v.to(device) for k, v in tpl_out.items()}

    # Case B ▸ plain models (Mistral, TinyLlama, etc.)
    plain_out = tokenizer(plain_prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in plain_out.items()}
