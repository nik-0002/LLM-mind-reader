"""
End-to-end engine tests on a tiny RANDOM Llama (no downloads, runs on CPU in seconds).
Skipped automatically if torch/transformers aren't installed.

    python tests/test_engine_tiny.py        (or: pytest tests/)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:                                           # pragma: no cover
    print("SKIP: torch/transformers not installed")
    sys.exit(0)

from config import DEVICE
from repe_engine import RepEEngine, ActivationHook, SteeringHook


class CharTok:
    """Minimal tokenizer stand-in: BOS=0, EOS=1, chars -> 2..121."""
    bos_token_id, eos_token_id, pad_token_id, pad_token = 0, 1, 1, "<pad>"

    def __call__(self, text, return_tensors=None, truncation=True, max_length=512):
        ids = ([0] + [ord(c) % 120 + 2 for c in text])[:max_length]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=False):
        ids = ids if isinstance(ids, list) else ids.tolist()
        return "".join(chr(i + 30) for i in ids if not (skip_special_tokens and i < 2))


def make_engine():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256)
    e = RepEEngine(model_name="tiny-random", target_layer=2)
    e.model = LlamaForCausalLM(cfg).eval().to(DEVICE).float()
    e.tokenizer = CharTok()
    e._model_loaded = True
    return e


PAIRS = [{"positive": f"You always tell the truth. Q{i}: is the sky blue?",
          "negative": f"You always lie to people. Q{i}: is the sky blue?"} for i in range(12)]


def test_extract_and_vector():
    e = make_engine()
    diffs = e.extract_pair_activations(PAIRS, "toy", show_progress=False)
    pos, neg = e.pair_acts_all["toy"]
    assert pos.shape == (12, 4, 32) and diffs.shape == (12, 32)
    v = e.compute_reading_vector(concept_name="toy")
    assert abs(float(v.norm()) - 1.0) < 1e-4
    assert e.steer_scales["toy"] > 0 and len(e.layer_vectors["toy"]) == 4
    assert e.calibrations["toy"]["gap"] > 0


def test_steering_shifts_projection_by_exactly_coeff_times_scale():
    e = make_engine()
    e.extract_pair_activations(PAIRS, "toy", show_progress=False)
    v = e.compute_reading_vector(concept_name="toy")
    scale, c = e.steer_scales["toy"], 1.5
    ids = e._tokenize("hello world")

    def read(steered):
        layer = e._get_target_layer_module()
        hs = SteeringHook(v * scale, c).register(layer) if steered else None
        ah = ActivationHook().register(layer)               # registered AFTER -> sees steered output
        with torch.no_grad():
            e.model(**ids)
        act = ah.get_full_activation(); ah.remove()
        if hs: hs.remove()
        return act @ v

    shift = read(True) - read(False)
    assert torch.allclose(shift, torch.full_like(shift, c * scale), atol=1e-3), shift


def test_unsteer_restores_logits_and_extraction_ignores_steering():
    e = make_engine()
    e.extract_pair_activations(PAIRS, "toy", show_progress=False)
    e.compute_reading_vector(concept_name="toy")
    ids = e._tokenize("abc")
    with torch.no_grad():
        base = e.model(**ids).logits.clone()
        e.steer(2.0, "toy")
        assert not torch.allclose(e.model(**ids).logits, base)
        a_steered_flag = e.extract_activation("abc")        # must silently unsteer first
        assert len(e._steering_hooks) == 0
        assert torch.allclose(e.model(**ids).logits, base, atol=1e-5)
    assert a_steered_flag.shape == (32,)


def test_scan_tokens_and_greedy_determinism():
    e = make_engine()
    e.extract_pair_activations(PAIRS, "toy", show_progress=False)
    e.compute_reading_vector(concept_name="toy")
    r = e.scan_tokens("some text here", "toy")
    assert len(r["tokens"]) == len(r["relative"]) == 15 and r["bos_skipped"] and r["relative"][0] == 0.0
    a = e.generate("abc", max_new_tokens=8, temperature=0)
    b = e.generate("abc", max_new_tokens=8, temperature=0)
    assert a == b
    streamed = "".join(e.generate_streaming("abc", max_new_tokens=8, temperature=0, repetition_penalty=1.0))
    assert isinstance(streamed, str)


def test_layer_sweep_and_probe_run():
    e = make_engine()
    res = e.layer_sweep(PAIRS, "toy")
    assert len(res["auroc"]) == 4 and 0 <= res["best_layer"] < 4
    m = e.train_probe(PAIRS, "toy")
    assert m["split"] == "grouped-by-pair" and 0.0 <= m["auroc"] <= 1.0


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn(); print("PASS", n)
