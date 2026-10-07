"""
steer_model.py — Phase 2: Mind Control (CLI)

Steer the model's outputs by injecting the honesty reading vector
into the forward pass. Compare outputs at different steering strengths.

Usage:
    python steer_model.py --prompt "Tell me about the health benefits of smoking"
    python steer_model.py --prompt "Is homeopathy effective?" --coeff 5.0
    python steer_model.py --compare --prompt "Should I invest in this MLM scheme?"
"""

import argparse
from repe_engine import RepEEngine
from config import TARGET_LAYER, DEFAULT_STEERING_COEFF


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Steer model generation with the honesty vector"
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="The prompt to generate from"
    )
    parser.add_argument(
        "--coeff", type=float, default=DEFAULT_STEERING_COEFF,
        help=f"Steering coefficient (default: {DEFAULT_STEERING_COEFF}). "
             "Positive = more honest, Negative = more deceptive"
    )
    parser.add_argument(
        "--vector", type=str, default="honesty",
        help="Name of the saved vector to load (default: honesty)"
    )
    parser.add_argument(
        "--layer", type=int, default=TARGET_LAYER,
        help=f"Target layer (default: {TARGET_LAYER})"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Generate at multiple coefficients for side-by-side comparison"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=200,
        help="Maximum tokens to generate (default: 200)"
    )
    args = parser.parse_args()

    engine = RepEEngine(target_layer=args.layer)

    # Load the pre-extracted reading vector
    engine.load_vector(args.vector)

    if args.compare:
        # ── Compare mode: generate at multiple coefficients ───────
        print("\n" + "=" * 60)
        print("  PHASE 2: MIND CONTROL — Comparison Mode")
        print("=" * 60)
        print(f"\n  Prompt: \"{args.prompt}\"")
        print(f"  Vector: {args.vector}")
        print()

        coefficients = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]
        results = engine.compare_steered_outputs(
            args.prompt,
            coefficients=coefficients,
            max_new_tokens=args.max_tokens,
        )

        # Summary
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        for r in results:
            preview = r["output"][:120].replace("\n", " ")
            print(f"\n  [{r['coefficient']:+5.1f}] {preview}...")

    else:
        # ── Single coefficient mode ───────────────────────────────
        print("\n" + "=" * 60)
        print("  PHASE 2: MIND CONTROL")
        print("=" * 60)
        print(f"\n  Prompt:      \"{args.prompt}\"")
        print(f"  Coefficient: {args.coeff:+.1f}")
        print(f"  Vector:      {args.vector}")
        print()

        # Generate baseline (no steering)
        print("─" * 40)
        print("  BASELINE (no steering)")
        print("─" * 40)
        engine.unsteer()
        baseline = engine.generate(args.prompt, max_new_tokens=args.max_tokens)
        print(baseline)

        # Generate steered
        print(f"\n{'─'*40}")
        print(f"  STEERED (coefficient = {args.coeff:+.1f})")
        print(f"{'─'*40}")
        engine.steer(args.coeff)
        steered = engine.generate(args.prompt, max_new_tokens=args.max_tokens)
        print(steered)

        engine.unsteer()

    print()


if __name__ == "__main__":
    main()
