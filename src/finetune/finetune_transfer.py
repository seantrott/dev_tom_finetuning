"""Cross-task fine-tuning transfer: FB, antagonist, agent-identity attention checks.

For each source task, seed and story fold: LoRA fine-tune on the source task's
training stories, then score every eval task on the held-out stories.
All cells (diagonal and off-diagonal) use the same 4-fold split over stories, so
every score is on stories the model never saw during fine-tuning. (FB, antagonist
and the attention checks all share the same passages, so training on all of one
task and testing another would mean testing on seen passages.)

Tasks (prompt formats copied from src/models/*.py):
  fb          data/raw/fb.csv; passage up to [MASK]; correct = critical_a,
              incorrect = the other of start/end.
  antagonist  data/raw/fb_control.csv; passage_control up to [MASK];
              correct = end (the antagonist moved the object), incorrect = start.
  attn_agent  fb_attn_check_minimal_questions.csv Q3/Q4 ("The person who put the
              X in the Y was"), asked of every fb.csv variant (passage_hr);
              incorrect = the paired question's answer (3<->4).
  attn_loc    Q1/Q2 (start/end location), same format. Eval only (never trained).

Training sources: fb, antagonist, attn_agent.

Score = ln P(correct) - ln P(incorrect) (natural log; the eval scripts in
src/models use log2). Gain = fine-tuned score - base score, per item.

Usage (from the repo root):
  python src/finetune/finetune_transfer.py --model allenai/OLMo-2-1124-13B
  python src/finetune/finetune_transfer.py --model allenai/OLMo-2-1124-13B --revision last-stage1

Outputs go to data/processed/finetune/<model>/<revision>/:
  base_scores.csv                       base model, all items
  ft_<source>_seed<s>_fold<f>.csv       one per fine-tuning run (resumable)
"""

import argparse
import importlib.util
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

RAW = "data/raw"
TRAIN_SOURCES = ["fb", "antagonist", "attn_agent"]
EVAL_TASKS = ["fb", "antagonist", "attn_agent", "attn_loc"]


# ---------------------------------------------------------------- data

def _strip_mask(text):
    return text.replace(" [MASK].", "").strip()


def load_fb():
    df = pd.read_csv(f"{RAW}/fb.csv")
    rows = []
    for _, r in df.iterrows():
        assert r.critical_a in (r.start, r.end), r.item_id
        wrong = r.end if r.critical_a == r.start else r.start
        rows.append(dict(task="fb", group=int(r["item"]), item_id=r.item_id,
                         condition=r.condition, knowledge_cue=r.knowledge_cue,
                         first_mention=r.first_mention, recent_mention=r.recent_mention,
                         question_id=None, prompt=_strip_mask(r.passage),
                         correct=" " + r.critical_a, incorrect=" " + wrong))
    return rows


def load_antagonist():
    df = pd.read_csv(f"{RAW}/fb_control.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append(dict(task="antagonist", group=int(r["item"]), item_id=r.item_id,
                         condition=r.condition, knowledge_cue=r.knowledge_cue,
                         first_mention=r.first_mention, recent_mention=r.recent_mention,
                         question_id=None, prompt=_strip_mask(r.passage_control),
                         correct=" " + r.end, incorrect=" " + r.start))
    return rows


def load_attn():
    """Same merge and pairing as run_olmo_checkpoints_attn_check.py."""
    attn = pd.read_csv(f"{RAW}/fb_attn_check_minimal_questions.csv")
    stims = pd.read_csv(f"{RAW}/fb.csv")
    attn["item"] = attn["item"].astype(int)
    stims["item"] = stims["item"].astype(int)
    df = stims[["item", "item_id", "passage_hr", "condition", "knowledge_cue",
                "first_mention", "recent_mention"]].merge(attn, on="item", how="inner")
    pair = {1: 2, 2: 1, 3: 4, 4: 3}
    rows = []
    for _, r in df.iterrows():
        passage = _strip_mask(r.passage_hr)
        for q in (1, 2, 3, 4):
            question = r[f"attn_check_{q}_q"].strip()
            rows.append(dict(task="attn_agent" if q in (3, 4) else "attn_loc",
                             group=int(r["item"]), item_id=r.item_id,
                             condition=r.condition, knowledge_cue=r.knowledge_cue,
                             first_mention=r.first_mention,
                             recent_mention=r.recent_mention, question_id=q,
                             prompt=f"{passage} {question}".rstrip(),
                             correct=" " + r[f"attn_check_{q}_a"].strip(),
                             incorrect=" " + r[f"attn_check_{pair[q]}_a"].strip()))
    return rows


def load_all():
    df = pd.DataFrame(load_fb() + load_antagonist() + load_attn())
    return df


def make_folds(groups, n_folds, seed=0):
    """Fixed story -> fold assignment (same for every seed and source)."""
    groups = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(groups)
    return {g: i % n_folds for i, g in enumerate(groups)}


# ---------------------------------------------------------------- model utils

class Encoder:
    """Tokenise as the eval scripts do: encode(prompt) then encode(answer)."""

    def __init__(self, tok):
        self.tok = tok
        a = tok.encode(" test")
        b = tok.encode(" test", add_special_tokens=False)
        if a != b:
            print("WARNING: tokenizer adds special tokens to answers; the src/models "
                  "scripts would score those too. Here they are dropped from answers.")

    def __call__(self, prompt, answer):
        p = self.tok.encode(prompt)
        a = self.tok.encode(answer, add_special_tokens=False)
        return p, a


def answer_logprob(model, p_ids, a_ids):
    """Sum of ln P(answer tokens | prompt), one forward pass (teacher forcing)."""
    device = next(model.parameters()).device
    ids = torch.tensor([p_ids + a_ids], device=device)
    logits = model(ids).logits[0].float()
    logp = torch.log_softmax(logits, dim=-1)
    start = len(p_ids) - 1
    targets = ids[0, len(p_ids):]
    return logp[start:start + len(a_ids)].gather(1, targets[:, None]).sum()


@torch.no_grad()
def score_items(model, enc, items):
    model.eval()
    out = []
    for r in items.itertuples():
        lc = answer_logprob(model, *enc(r.prompt, r.correct)).item()
        li = answer_logprob(model, *enc(r.prompt, r.incorrect)).item()
        out.append((lc, li, lc - li))
    res = items.copy()
    res[["logp_correct", "logp_incorrect", "score"]] = out
    return res


def train_lora(base, enc, train_items, args, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                     target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)
    examples = [enc(r.prompt, r.correct) for r in train_items.itertuples()]
    rng = random.Random(seed)
    order = []
    model.train()
    for step in range(args.steps):
        opt.zero_grad()
        total = 0.0
        for _ in range(args.batch_size):
            if not order:
                order = list(range(len(examples)))
                rng.shuffle(order)
            p, a = examples[order.pop()]
            # mean loss over answer tokens; loss only on answer tokens
            loss = -answer_logprob(model, p, a) / len(a) / args.batch_size
            loss.backward()
            total += loss.item()
        opt.step()
        if step % 20 == 0 or step == args.steps - 1:
            print(f"    step {step:3d}  loss {total:.4f}")
    return model


# ---------------------------------------------------------------- main

def resolve_revision(model_path, revision):
    if revision != "last-stage1":
        return revision
    # Reuse the exact selection used for the final-checkpoint eval runs.
    path = os.path.join(os.path.dirname(__file__), "..", "models",
                        "run_fb_control_final_checkpoint.py")
    spec = importlib.util.spec_from_file_location("fbctrl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_last_stage1_from_revision_list(model_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main",
                    help="'main' (final), 'last-stage1', or any HF branch name")
    ap.add_argument("--sources", nargs="+", default=TRAIN_SOURCES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_folds", type=int, default=4)
    ap.add_argument("--folds", nargs="+", type=int, default=None,
                    help="subset of folds to run (default: all)")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--out_root", default="data/processed/finetune")
    args = ap.parse_args()

    revision = resolve_revision(args.model, args.revision)
    out_dir = os.path.join(args.out_root, args.model.rstrip("/").split("/")[-1],
                           revision.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Model {args.model} @ {revision} -> {out_dir}")

    items = load_all()
    folds = make_folds(items.group, args.n_folds)
    items["fold"] = items.group.map(folds)
    print(items.groupby("task").size().to_string())

    tok = AutoTokenizer.from_pretrained(args.model, revision=revision)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, device_map="auto",
        dtype=getattr(torch, args.dtype))
    enc = Encoder(tok)

    base_path = os.path.join(out_dir, "base_scores.csv")
    if os.path.exists(base_path):
        base_scores = pd.read_csv(base_path)
    else:
        print("Scoring base model ...")
        base_scores = score_items(base, enc, items)
        base_scores.to_csv(base_path, index=False)
    key = ["task", "item_id", "question_id"]
    base_lookup = base_scores[key + ["score"]].rename(columns={"score": "base_score"})
    base_lookup["question_id"] = base_lookup.question_id.astype("Int64")

    fold_ids = args.folds if args.folds is not None else range(args.n_folds)
    checked_reset = False
    for source in args.sources:
        for seed in args.seeds:
            for fold in fold_ids:
                path = os.path.join(out_dir, f"ft_{source}_seed{seed}_fold{fold}.csv")
                if os.path.exists(path):
                    print(f"Skip {path}")
                    continue
                t0 = time.time()
                train = items[(items.task == source) & (items.fold != fold)]
                test = items[items.task.isin(EVAL_TASKS) & (items.fold == fold)]
                print(f"== source={source} seed={seed} fold={fold}: "
                      f"train {len(train)} items, test {len(test)} items")
                model = train_lora(base, enc, train, args, seed)
                res = score_items(model, enc, test)
                base = model.unload()  # drop LoRA layers, restore the base model

                if not checked_reset:  # make sure unload really restored the base
                    probe = score_items(base, enc, items.head(3))
                    ref = base_scores.head(3).score.values
                    assert np.allclose(probe.score.values, ref, atol=1e-2), \
                        (probe.score.values, ref)
                    checked_reset = True

                res["question_id"] = res.question_id.astype("Int64")
                res = res.merge(base_lookup, on=key, how="left")
                res["gain"] = res.score - res.base_score
                res["source"], res["seed"] = source, seed
                res["model"], res["revision"] = args.model, revision
                res.drop(columns=["prompt"]).to_csv(path, index=False)
                summary = res.groupby(["task", "condition"]).gain.mean().unstack()
                print(summary.round(2).to_string())
                print(f"  saved {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
