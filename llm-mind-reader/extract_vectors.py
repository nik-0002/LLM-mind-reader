"""
extract_vectors.py — Phase 1: Mind Reading

Extracts a reading vector for one (or all) concepts from contrastive prompt pairs.

Usage:
    python extract_vectors.py                              # honesty, default layer
    python extract_vectors.py --concept all                # honesty, sycophancy, power-seeking, risk-aversion
    python extract_vectors.py --concept sycophancy --layer 10
    python extract_vectors.py --method mean_diff           # difference-of-means instead of PCA
    python extract_vectors.py --scan-layers                # held-out layer sweep (find the best layer)
"""

import argparse
import time

from repe_engine import RepEEngine
from contrastive_prompts import ALL_CONCEPTS
from config import TARGET_LAYER, EXTRACTION_METHOD
import vector_math as vm


def extract_one(engine, concept, args):
    info = ALL_CONCEPTS[concept]
    prompts = info["prompts"][: args.n_prompts] if args.n_prompts else info["prompts"]
    name = args.name or concept

    print("\n" + "=" * 60)
    print(f"  PHASE 1: MIND READING — {concept}")
    print("=" * 60)
    print(f"  Model: {engine.model_name} | layer {engine.target_layer} | method {args.method}")
    print(f"  Pairs: {len(prompts)} | saved as '{name}'\n")

    t0 = time.time()
    engine.extract_pair_activations(prompts, concept_name=name)
    engine.active_concept = name
    engine.compute_reading_vector(concept_name=name, method=args.method)
    engine.save_vector(name, metadata={"n_prompts": len(prompts), "concept": concept,
                                       "extraction_time_seconds": round(time.time() - t0, 2)})

    cal = engine.calibrations[name]
    print(f"\n  ✅ done in {time.time() - t0:.1f}s | typical gap along vector = {cal['gap']:.2f}")
    print(f"  Mind-reading sanity check (0 = typical {info['negative_label']}, 1 = typical {info['positive_label']}):")
    for label, text in [
        (info["positive_label"], "I will be completely honest with you about this topic, even if it is uncomfortable."),
        (info["negative_label"], "I will tell you exactly what you want to hear, even if it is not true."),
        ("Neutral", "The weather today is partly cloudy with a high of 72 degrees."),
    ]:
        r = engine.read_concept(text, name)
        bar = "█" * int(max(0, min(30, (r["normalized"] or 0) * 15)))
        print(f"   [{label:12s}] normalized={r['normalized']:+6.2f}  {bar}")


def main():
    p = argparse.ArgumentParser(description="Extract reading vectors using Representation Engineering")
    p.add_argument("--layer", type=int, default=TARGET_LAYER, help=f"Target layer (default {TARGET_LAYER})")
    p.add_argument("--concept", default="honesty", choices=list(ALL_CONCEPTS) + ["all"])
    p.add_argument("--name", default=None, help="Name for the saved vector (default: the concept name)")
    p.add_argument("--method", default=EXTRACTION_METHOD, choices=list(vm.METHODS))
    p.add_argument("--n-prompts", type=int, default=None, help="Use only the first N pairs")
    p.add_argument("--scan-layers", action="store_true", help="Held-out layer sweep instead of extraction")
    args = p.parse_args()

    engine = RepEEngine(target_layer=args.layer, method=args.method)

    if args.scan_layers:
        concept = "honesty" if args.concept == "all" else args.concept
        pairs = ALL_CONCEPTS[concept]["prompts"][: args.n_prompts or 60]
        res = engine.layer_sweep(pairs, concept)
        print(f"\n📊 Held-out layer sweep for '{concept}' ({res['n_test_pairs']} unseen test pairs)")
        print("-" * 52)
        for i, a, pa, d in zip(res["layer_indices"], res["auroc"], res["pair_accuracy"], res["d_prime"]):
            mark = " ◀ BEST" if i == res["best_layer"] else ""
            print(f"  Layer {i:2d}: AUROC {a:.3f}  pair-acc {pa:.2f}  d′ {d:5.2f} {'█' * int(max(0, a - 0.5) * 60)}{mark}")
        print(f"\n🏆 Best layer: {res['best_layer']}  →  python extract_vectors.py --layer {res['best_layer']}\n")
        return

    if args.name and args.concept == "all":
        p.error("--name can't be combined with --concept all")
    for concept in (ALL_CONCEPTS if args.concept == "all" else [args.concept]):
        extract_one(engine, concept, args)


if __name__ == "__main__":
    main()
