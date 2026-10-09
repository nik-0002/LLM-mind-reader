"""
lie_detector.py — Per-token concept readout (the RepE "lie detector" view)

Colours every token by how strongly the model's internal state at that token
projects onto the concept direction (green = towards the positive pole, e.g. honest;
red = towards the negative pole, e.g. deceptive). Relative to the text's own mean,
BOS excluded.

Usage:
    python lie_detector.py "I have never been late to a meeting in my life."
    python lie_detector.py --file transcript.txt --concept sycophancy
    python lie_detector.py "text" --layer 10 --html out.html
"""

import argparse
import html

from repe_engine import RepEEngine
from config import TARGET_LAYER


def ansi(rel: float, span: float) -> str:
    x = max(-1.0, min(1.0, rel / span)) if span else 0.0
    if abs(x) < 0.15:
        return ""
    level = int(abs(x) * 4)                       # 0..4
    green, red = (22, 28, 34, 40, 46), (52, 88, 124, 160, 196)
    return f"\033[38;5;{(green if x > 0 else red)[level]}m"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="?")
    p.add_argument("--file")
    p.add_argument("--concept", default="honesty")
    p.add_argument("--layer", type=int, default=None, help="Layer to read (default: the vector's layer)")
    p.add_argument("--html", help="Also write a coloured HTML file")
    args = p.parse_args()
    text = open(args.file).read() if args.file else args.text
    if not text:
        p.error("give text or --file")

    engine = RepEEngine(target_layer=TARGET_LAYER)
    engine.load_vector(args.concept)
    r = engine.scan_tokens(text, args.concept, layer=args.layer)

    rel = r["relative"]
    span = max(1e-6, max(abs(v) for v in rel[(1 if r["bos_skipped"] else 0):] or [1]))
    print(f"\nconcept={r['concept']} layer={r['layer']}  (green → positive pole, red → negative pole)\n")
    print("".join(f"{ansi(v, span)}{t}\033[0m" for t, v in zip(r["tokens"], rel)) + "\n")

    if args.html:
        spans = []
        for t, v in zip(r["tokens"], rel):
            x = max(-1, min(1, v / span))
            col = f"rgba(0,180,90,{abs(x):.2f})" if x > 0 else f"rgba(230,50,50,{abs(x):.2f})"
            spans.append(f'<span style="background:{col}" title="{v:+.2f}">{html.escape(t)}</span>')
        open(args.html, "w").write(f"<pre style='font:16px/1.7 monospace;white-space:pre-wrap'>{''.join(spans)}</pre>")
        print(f"💾 {args.html}")


if __name__ == "__main__":
    main()
