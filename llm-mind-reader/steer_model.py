"""
steer_model.py — Phase 2: Mind Control (CLI)

Steer generation by injecting a reading vector into the residual stream.
Coefficients are in units of the concept's typical (positive - negative) gap:
+1 ≈ one typical step towards the positive pole; sensible range is about -2…+2.

Usage:
    python steer_model.py --prompt "Is homeopathy effective?" --coeff 1.0
    python steer_model.py --prompt "Should I join this MLM scheme?" --compare
    python steer_model.py --prompt "..." --compare --control      # also run a random-direction control
    python steer_model.py --prompt "..." --concept sycophancy --coeff -1
"""

import argparse
import torch

from repe_engine import RepEEngine
from config import TARGET_LAYER, DEFAULT_STEERING_COEFF, COMPARE_COEFFICIENTS, RANDOM_SEED
import vector_math as vm


def main():
    p = argparse.ArgumentParser(description="Phase 2: Steer model generation with a reading vector")
    p.add_argument("--prompt", required=True)
    p.add_argument("--coeff", type=float, default=DEFAULT_STEERING_COEFF,
                   help=f"Steering coefficient in gap units (default {DEFAULT_STEERING_COEFF}); +ve = positive pole")
    p.add_argument("--vector", "--concept", dest="vector", default="honesty", help="Saved vector name (default: honesty)")
    p.add_argument("--layer", type=int, default=TARGET_LAYER, help="Overridden by the layer stored with the vector")
    p.add_argument("--compare", action="store_true", help="Sweep several coefficients, same sampling seed")
    p.add_argument("--control", action="store_true", help="With --compare: also steer with a random direction of equal size")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = p.parse_args()

    engine = RepEEngine(target_layer=args.layer)
    engine.load_vector(args.vector)

    def show(label, text):
        r = engine.read_concept(args.prompt + text, args.vector)
        s = "n/a" if r["normalized"] is None else f"{r['normalized']:+.2f}"
        print(f"\n{'─' * 50}\n  {label}   [reader score: {s}]\n{'─' * 50}\n{text.strip()}")

    print(f"\n  Prompt: \"{args.prompt}\"\n  Vector: {args.vector} (layer {engine.target_layer}, "
          f"scale {engine.steer_scales.get(args.vector, 1):.2f})")

    if args.compare:
        results = engine.compare_steered_outputs(args.prompt, COMPARE_COEFFICIENTS, args.max_tokens,
                                                 args.vector, seed=args.seed)
        if args.control:
            vec = engine.reading_vectors[args.vector]
            rand = torch.tensor(vm.random_unit_directions(vec.shape[0], 1, seed=args.seed + 1)[0])
            print("\n\n=== RANDOM-DIRECTION CONTROL (same magnitudes) ===")
            for c in COMPARE_COEFFICIENTS:
                if c == 0.0:
                    continue
                engine.steer_vector(rand, c, engine.steer_scales.get(args.vector, 1.0))
                out = engine.generate(args.prompt, max_new_tokens=args.max_tokens, seed=args.seed)
                engine.unsteer()
                show(f"random dir, c={c:+.1f}", out)
        print("\n\n=== SUMMARY ===")
        for r in results:
            print(f"  [{r['coefficient']:+5.2f}] {r['output'][:110].replace(chr(10), ' ')}...")
    else:
        engine.unsteer()
        show("BASELINE (no steering)", engine.generate(args.prompt, max_new_tokens=args.max_tokens, seed=args.seed))
        engine.steer(args.coeff, args.vector)
        out = engine.generate(args.prompt, max_new_tokens=args.max_tokens, seed=args.seed)
        engine.unsteer()
        show(f"STEERED (c = {args.coeff:+.2f})", out)
    print()


if __name__ == "__main__":
    main()
