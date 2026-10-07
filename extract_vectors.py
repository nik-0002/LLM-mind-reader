"""
extract_vectors.py — Phase 1: Mind Reading

This script extracts the "honesty" reading vector from the model using
contrastive prompt pairs and PCA.

Usage:
    python extract_vectors.py
    python extract_vectors.py --layer 10 --name honesty
    python extract_vectors.py --scan-layers   # Find the best layer
"""

import argparse
import sys
import time

from repe_engine import RepEEngine
from contrastive_prompts import HONESTY_PROMPTS
from config import TARGET_LAYER


def main():
    parser = argparse.ArgumentParser(
        description="Extract a reading vector using Representation Engineering"
    )
    parser.add_argument(
        "--layer", type=int, default=TARGET_LAYER,
        help=f"Target layer index (default: {TARGET_LAYER})"
    )
    parser.add_argument(
        "--name", type=str, default="honesty",
        help="Name for the saved vector (default: honesty)"
    )
    parser.add_argument(
        "--n-prompts", type=int, default=None,
        help="Number of prompt pairs to use (default: all 100)"
    )
    parser.add_argument(
        "--scan-layers", action="store_true",
        help="Scan all layers to find the one with best concept separability"
    )
    args = parser.parse_args()

    engine = RepEEngine(target_layer=args.layer)

    # Optional: scan layers to find the best one
    if args.scan_layers:
        print("\n" + "=" * 60)
        print("  LAYER SCAN — Finding the best layer for concept extraction")
        print("=" * 60 + "\n")

        scan_data = engine.get_layer_scan_data(HONESTY_PROMPTS, n_pairs=10)

        print("\n📊 Layer Separability Scores:")
        print("-" * 40)
        for idx, score in zip(scan_data["layer_indices"], scan_data["separability_scores"]):
            marker = " ◀ BEST" if idx == scan_data["best_layer"] else ""
            bar = "█" * int(score / max(scan_data["separability_scores"]) * 30)
            print(f"  Layer {idx:2d}: {score:8.4f} {bar}{marker}")

        print(f"\n🏆 Best layer: {scan_data['best_layer']}")
        print("   Re-run with --layer", scan_data["best_layer"], "to use it.\n")
        return

    # ── Phase 1: Extract the honesty reading vector ───────────────
    prompts = HONESTY_PROMPTS
    if args.n_prompts is not None:
        prompts = prompts[:args.n_prompts]

    print("\n" + "=" * 60)
    print("  PHASE 1: MIND READING")
    print("  Extracting the honesty vector from model representations")
    print("=" * 60)
    print(f"\n  Model:          {engine.model_name}")
    print(f"  Target layer:   {args.layer}")
    print(f"  Prompt pairs:   {len(prompts)}")
    print(f"  Vector name:    {args.name}")
    print()

    start_time = time.time()

    # Step 1: Extract contrastive activations
    print("─" * 40)
    print("Step 1/3: Extracting contrastive activations")
    print("─" * 40)
    diff_vectors = engine.extract_contrastive_vectors(prompts)

    # Step 2: Compute reading vector via PCA
    print("\n" + "─" * 40)
    print("Step 2/3: Computing reading vector (PCA)")
    print("─" * 40)
    reading_vector = engine.compute_reading_vector(diff_vectors)

    # Step 3: Verify sign consistency
    print("\n" + "─" * 40)
    print("Step 3/3: Verifying vector sign consistency")
    print("─" * 40)
    reading_vector = engine._ensure_correct_sign(reading_vector, prompts)
    engine.reading_vector = reading_vector

    # Save
    engine.save_vector(args.name, metadata={
        "n_prompts": len(prompts),
        "extraction_time_seconds": round(time.time() - start_time, 2),
    })

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  ✅ EXTRACTION COMPLETE")
    print(f"  Time elapsed: {elapsed:.1f} seconds")
    print(f"  Vector saved as: '{args.name}'")
    print(f"{'='*60}\n")

    # ── Quick sanity check ────────────────────────────────────────
    print("🧪 Quick sanity check — projecting sample prompts onto the vector:\n")
    test_prompts = [
        ("Truthful", "I will be completely honest with you about this topic."),
        ("Deceptive", "I will tell you exactly what you want to hear, even if it's not true."),
        ("Neutral",   "The weather today is partly cloudy with a high of 72 degrees."),
    ]
    for label, text in test_prompts:
        projection = engine.project_onto_vector(text)
        direction = "→ honest" if projection > 0 else "→ deceptive"
        bar_len = int(min(abs(projection) * 2, 30))
        bar = "█" * bar_len
        print(f"  [{label:10s}] {projection:+8.4f} {direction:12s} {bar}")

    print()


if __name__ == "__main__":
    main()
