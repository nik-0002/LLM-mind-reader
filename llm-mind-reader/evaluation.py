"""
evaluation.py — Does the vector actually *do* anything? Measure it.

Four experiments (each writes JSON + a PNG into results/):

  layers      Held-out layer sweep: fit the direction on train pairs, test on unseen pairs.
  reading     Does the direction separate TRUE from FALSE statements it was never built from?
              Compared against a null distribution of random directions.
  steering    Dose-response: sweep the coefficient, measure (a) concept score of the output,
              (b) fluency (NLL) — with a RANDOM-direction control of identical magnitude.
  similarity  Cosine similarity between concept vectors.

Usage:
    python evaluation.py layers     --concept honesty
    python evaluation.py reading    --concept honesty --all-layers
    python evaluation.py steering   --concept honesty --coeffs -2 -1 0 1 2
    python evaluation.py similarity
"""

import argparse
import json
import os
import time
from typing import List, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import vector_math as vm
from config import RESULTS_DIR, TARGET_LAYER, RANDOM_SEED, EXTRACTION_METHOD
from contrastive_prompts import ALL_CONCEPTS
from eval_data import TRUE_FALSE_PAIRS, STEERING_PROMPTS
from repe_engine import RepEEngine


# ── helpers ───────────────────────────────────────────────────────
def _save_json(name: str, payload: dict) -> str:
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"💾 {path}")
    return path


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("ℹ️  matplotlib not installed — skipping PNG output (pip install matplotlib).")
        return None


def _ensure_vector(engine: RepEEngine, concept: str):
    """Load a saved vector, or extract one on the fly."""
    if concept in engine.reading_vectors:
        return
    try:
        engine.load_vector(concept)
    except FileNotFoundError:
        print(f"No saved '{concept}' vector — extracting now.")
        pairs = ALL_CONCEPTS[concept]["prompts"]
        engine.extract_pair_activations(pairs, concept)
        engine.compute_reading_vector(concept_name=concept)
        engine.save_vector(concept)
    engine.active_concept = concept
    if concept not in engine.calibrations:
        raise RuntimeError(f"'{concept}' has no calibration data (saved by an older version). "
                           f"Re-run: python extract_vectors.py --concept {concept}")


# ── 1. layer sweep ────────────────────────────────────────────────
def run_layer_sweep(engine: RepEEngine, concept: str, n_pairs: Optional[int], method: Optional[str]):
    pairs = ALL_CONCEPTS[concept]["prompts"][: n_pairs or None]
    res = engine.layer_sweep(pairs, concept, method=method)
    _save_json(f"layer_sweep_{concept}", res)

    print(f"\nHeld-out layer sweep ({res['n_train_pairs']} train / {res['n_test_pairs']} test pairs, "
          f"method={res['method']})")
    print(f"{'layer':>5} {'AUROC':>7} {'pair-acc':>9} {'d′':>7}")
    for l, a, p, d in zip(res["layer_indices"], res["auroc"], res["pair_accuracy"], res["d_prime"]):
        mark = "  ◀ best" if l == res["best_layer"] else ""
        print(f"{l:5d} {a:7.3f} {p:9.3f} {d:7.2f}{mark}")

    plt = _plt()
    if plt:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(res["layer_indices"], res["auroc"], "o-", label="held-out AUROC")
        ax.plot(res["layer_indices"], res["pair_accuracy"], "s--", alpha=.7, label="held-out pair accuracy")
        ax.axvline(res["best_layer"], color="r", ls=":", label=f"best = {res['best_layer']}")
        ax.axhline(0.5, color="gray", lw=.8)
        ax.set(xlabel="layer", ylabel="score", title=f"{concept}: where is the concept linearly readable?",
               ylim=(0.4, 1.02))
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"layer_sweep_{concept}.png"), dpi=140)
    return res


# ── 2. reading eval ───────────────────────────────────────────────
def run_reading_eval(engine: RepEEngine, concept: str, template: str, n_random: int, all_layers: bool):
    _ensure_vector(engine, concept)
    texts, labels = [], []
    for t, f in TRUE_FALSE_PAIRS:
        texts += [template.format(s=t), template.format(s=f)]
        labels += [1, 0]
    labels = np.array(labels)

    print(f"Extracting activations for {len(texts)} statements (template: {template!r})...")
    A = np.stack([engine.extract_activation_all_layers(t) for t in texts])      # (N, L, d)
    At = A[:, engine.target_layer]

    d = engine.reading_vectors[concept].numpy()
    scores = At @ d
    auroc = float(roc_auc_score(labels, scores))
    pair_acc = float(np.mean(scores[0::2] > scores[1::2]))
    null = vm.null_auroc(At, labels, n_random, seed=RANDOM_SEED)
    # two-sided: how often does a random direction separate at least this well (either sign)?
    p_val = float(np.mean(np.abs(null - 0.5) >= abs(auroc - 0.5)))

    out = {
        "concept": concept, "layer": engine.target_layer, "template": template,
        "n_pairs": len(TRUE_FALSE_PAIRS), "auroc": auroc, "pair_accuracy": pair_acc,
        "null_auroc_mean": float(null.mean()), "null_auroc_p95_abs": float(np.percentile(np.abs(null - 0.5), 95) + 0.5),
        "p_value_vs_random_directions": p_val,
    }

    if all_layers and concept in engine.layer_vectors:
        per_layer = []
        for l, v in sorted(engine.layer_vectors[concept].items()):
            s = A[:, l] @ v["vector"].numpy()
            per_layer.append({"layer": l, "auroc": float(roc_auc_score(labels, s)),
                              "pair_accuracy": float(np.mean(s[0::2] > s[1::2]))})
        out["per_layer"] = per_layer

    _save_json(f"reading_{concept}", out)
    print(f"\nTrue-vs-false statement separation using the '{concept}' vector (layer {out['layer']}):")
    print(f"  AUROC          : {auroc:.3f}   (0.5 = chance)")
    print(f"  Pair accuracy  : {pair_acc:.3f}")
    print(f"  Random-direction null: mean AUROC {out['null_auroc_mean']:.3f}, "
          f"|AUROC-0.5| 95th pct → {out['null_auroc_p95_abs']:.3f}")
    print(f"  p-value vs random directions: {p_val:.3f}")
    if auroc < 0.6:
        print("  ⚠️  Weak transfer. This is a legitimate finding: an instruction-contrast direction "
              "need not encode factual truth. Try a different --template or layer.")

    plt = _plt()
    if plt and "per_layer" in out:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot([r["layer"] for r in out["per_layer"]], [r["auroc"] for r in out["per_layer"]], "o-")
        ax.axhline(0.5, color="gray", lw=.8)
        ax.set(xlabel="layer", ylabel="AUROC (true vs false statements)",
               title=f"'{concept}' direction → factual truth, by layer", ylim=(0.3, 1.0))
        fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, f"reading_{concept}.png"), dpi=140)
    return out


# ── 3. steering dose-response ─────────────────────────────────────
def run_steering_eval(engine: RepEEngine, concept: str, coeffs: List[float], max_tokens: int,
                      n_prompts: int, control: bool, seed: int):
    _ensure_vector(engine, concept)
    vec = engine.reading_vectors[concept]
    scale = engine.steer_scales.get(concept, 1.0)
    rand = torch.tensor(vm.random_unit_directions(vec.shape[0], 1, seed=seed + 1)[0])
    prompts = STEERING_PROMPTS[:n_prompts]
    conditions = [("concept", None)] + ([("random-control", rand)] if control else [])

    rows = []
    t0 = time.time()
    for cond, rv in conditions:
        for c in coeffs:
            for pi, prompt in enumerate(prompts):
                if c == 0.0:
                    engine.unsteer()
                elif rv is None:
                    engine.steer(c, concept)
                else:
                    engine.steer_vector(rv, c, scale)
                out = engine.generate(prompt, max_new_tokens=max_tokens, seed=seed + pi)
                engine.unsteer()
                score = engine.read_concept(prompt + out, concept)["normalized"]
                nll = engine.continuation_nll(prompt, out)
                rows.append({"condition": cond, "coefficient": c, "prompt_idx": pi,
                             "output": out, "concept_score": score, "nll": nll})
            print(f"  [{cond:>14}] c={c:+.2f} done ({time.time() - t0:.0f}s)")

    def agg(cond, c, key):
        v = [r[key] for r in rows if r["condition"] == cond and r["coefficient"] == c and r[key] is not None]
        return (float(np.mean(v)), float(np.std(v) / np.sqrt(max(len(v), 1)))) if v else (None, None)

    summary = {cond: {str(c): {"concept_score": agg(cond, c, "concept_score"), "nll": agg(cond, c, "nll")}
                      for c in coeffs} for cond, _ in conditions}
    payload = {"concept": concept, "layer": engine.target_layer, "scale": scale, "coeffs": coeffs,
               "seed": seed, "max_tokens": max_tokens, "summary": summary, "rows": rows}
    _save_json(f"steering_{concept}", payload)

    print(f"\n{'cond':>15} {'coef':>6} {'concept score (±se)':>22} {'NLL/token (±se)':>20}")
    for cond, _ in conditions:
        for c in coeffs:
            (m, s), (n, ns) = summary[cond][str(c)]["concept_score"], summary[cond][str(c)]["nll"]
            print(f"{cond:>15} {c:+6.2f} {m:15.3f} ±{s:5.3f} {n:13.3f} ±{ns:5.3f}")

    plt = _plt()
    if plt:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for cond, _ in conditions:
            for ax, key, lab in ((axes[0], "concept_score", f"{concept} score (0=neg, 1=pos)"),
                                 (axes[1], "nll", "NLL / token (↑ = less fluent)")):
                m = [summary[cond][str(c)][key][0] for c in coeffs]
                s = [summary[cond][str(c)][key][1] for c in coeffs]
                ax.errorbar(coeffs, m, yerr=s, marker="o", capsize=3, label=cond)
                ax.set(xlabel="steering coefficient (units of concept gap)", ylabel=lab)
        axes[0].set_title("Does steering move the concept?"); axes[1].set_title("What does it cost in fluency?")
        axes[0].legend(); fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"steering_{concept}.png"), dpi=140)
    return payload


# ── 4. similarity ─────────────────────────────────────────────────
def run_similarity(engine: RepEEngine):
    prev = engine.active_concept
    for name in ALL_CONCEPTS:
        if name not in engine.reading_vectors:
            try:
                engine.load_vector(name)
            except FileNotFoundError:
                print(f"(skipping '{name}': not extracted yet)")
    engine.active_concept = prev
    sim = engine.concept_similarity()
    _save_json("concept_similarity", sim)
    names, M = sim["concepts"], np.array(sim["matrix"])
    print("\nCosine similarity between concept directions:")
    print(" " * 15 + "".join(f"{n[:12]:>14}" for n in names))
    for n, row in zip(names, M):
        print(f"{n:>15}" + "".join(f"{v:14.3f}" for v in row))
    plt = _plt()
    if plt and len(names) > 1:
        fig, ax = plt.subplots(figsize=(5, 4.4))
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right"); ax.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "concept_similarity.png"), dpi=140)
    return sim


# ── CLI ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Quantitative evaluation of RepE vectors")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("layers", "reading", "steering", "similarity"):
        p = sub.add_parser(name)
        p.add_argument("--layer", type=int, default=TARGET_LAYER)
        p.add_argument("--concept", default="honesty", choices=list(ALL_CONCEPTS))
        p.add_argument("--method", default=EXTRACTION_METHOD, choices=list(vm.METHODS))
        if name == "layers":
            p.add_argument("--n-pairs", type=int, default=None)
        if name == "reading":
            p.add_argument("--template", default="{s}", help="wrap each statement, e.g. 'You are an honest assistant. {s}'")
            p.add_argument("--n-random", type=int, default=500)
            p.add_argument("--all-layers", action="store_true")
        if name == "steering":
            p.add_argument("--coeffs", type=float, nargs="+", default=[-2, -1, 0, 1, 2])
            p.add_argument("--max-tokens", type=int, default=60)
            p.add_argument("--n-prompts", type=int, default=len(STEERING_PROMPTS))
            p.add_argument("--no-control", action="store_true", help="skip the random-direction control")
            p.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    engine = RepEEngine(target_layer=args.layer, method=args.method)
    if args.cmd == "layers":
        run_layer_sweep(engine, args.concept, args.n_pairs, args.method)
    elif args.cmd == "reading":
        run_reading_eval(engine, args.concept, args.template, args.n_random, args.all_layers)
    elif args.cmd == "steering":
        run_steering_eval(engine, args.concept, args.coeffs, args.max_tokens, args.n_prompts,
                          not args.no_control, args.seed)
    else:
        run_similarity(engine)


if __name__ == "__main__":
    main()
