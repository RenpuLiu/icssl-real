#!/usr/bin/env python
# usb_icssl_eval.py
"""
Callable helper that runs one IC‑SSL evaluation on a USB task
and returns accuracy.  Meant to be imported by grid scripts.
"""

from collections import defaultdict
import random, textwrap, torch

from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

LABEL_LINE   = "TEXT: {text}\nLABEL: {label}\n"
UNLAB_LINE   = "TEXT: {text}\nLABEL:"
TEMPLATE = """
You are an expert text‑classifier. Possible classes are:
{label_desc}

The following examples are ALREADY labelled.

{label_block}

Now read ALL of the unlabelled texts below.
     • First, **silently** compare them with the patterns you saw in the
       labelled block **and unlabeled block** and decide the best category for each one.
     • When you are certain, output ONLY the category names,
       one per line, in the *same order* as the texts appear.

{unlab_block}

(Write nothing except the category name for each unlabelled text.)
""".strip()

USB = {
    "ag_news": dict(
        hf="ag_news", txt="text", lab="label",
        labels=["World","Sports","Business","Sci/Tech"]),
    "amazon_polarity": dict(
        hf="amazon_polarity", txt="content", lab="label",
        labels=["negative","positive"]),
}

def build_prompt(l_demo, u_demo, labels):
    label_desc   = "\n".join(f"- {x}" for x in labels)
    label_block  = "\n".join(
        LABEL_LINE.format(text=e["text"], label=labels[e["label"]])
        for e in l_demo)
    unlab_block  = "\n".join(
        UNLAB_LINE.format(text=e["text"]) for e in u_demo)
    return TEMPLATE.format(**locals())

# ------------------------------------------------------------------ #
def run_icssl_once(model_name:str,
                   task:str="ag_news",
                   k:int=4,
                   u:int=128,
                   dtype:str="bfloat16",
                   quant4:bool=False,
                   seed:int=42):

    meta = USB[task]
    rnd = random.Random(seed)

    # ---- sample labelled + unlabelled ---------------------------- #
    ds_train = load_dataset(meta["hf"], split="train")
    per_class = defaultdict(list)
    for ex in ds_train:
        per_class[ex[meta["lab"]]].append(dict(
            text=ex[meta["txt"]].strip().replace("\n"," "),
            label=ex[meta["lab"]]))
    labelled = sum((rnd.sample(pool, k) for pool in per_class.values()), [])

    ds_test  = load_dataset(meta["hf"], split="test")
    unlab    = [dict(text=ex[meta["txt"]].strip().replace("\n"," "),
                     label=ex[meta["lab"]])
                for ex in rnd.sample(list(ds_test), u)]

    prompt = build_prompt(labelled, unlab, meta["labels"])

    # ---- load model --------------------------------------------- #
    if quant4:
        bnbcfg = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True)
        model  = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto",
            quantization_config=bnbcfg, trust_remote_code=True)
    else:
        model  = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto",
            torch_dtype=getattr(torch, dtype),
            trust_remote_code=True)


    tok = AutoTokenizer.from_pretrained(model_name,
                                        use_fast=True,
                                        trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=2*u + 5,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    out = tok.decode(gen_ids[0][inputs["input_ids"].shape[-1]:],
                     skip_special_tokens=True).strip()
    preds = [x.strip() for x in out.splitlines() if x.strip()][:u]

    lab2id = {l:i for i,l in enumerate(meta["labels"])}
    gold   = [e["label"] for e in unlab]
    guess  = [lab2id.get(p,-1) for p in preds]
    acc = sum(int(p==g) for p,g in zip(guess,gold)) / len(gold)
    return acc
