"""
repe_engine.py — Core Representation Engineering Engine (v3.0)

Pipeline:
  1. Activation extraction via PyTorch forward hooks (all layers in ONE pass)
  2. Contrastive difference computation
  3. Direction isolation: RepE-style PCA (with pair-order randomisation) or difference-of-means
  4. Calibrated steering: coefficient is in units of the concept's typical pos-neg gap
  5. Per-layer vectors -> multi-layer steering with layer-appropriate directions
  6. Linear probes with pair-grouped train/test splits
  7. Held-out layer sweep (replaces the old "norm of difference" scan, which just
     grows with depth and says nothing about separability)
  8. Per-token concept scan ("lie detector" view) + concept similarity + activation scatter

New in v3 vs v2 (see README "What changed"):
  - Steering was effectively a no-op: the PCA vector has unit norm, residual
    streams have norms of tens-hundreds. Vectors now carry a `steer_scale`.
  - Centered PCA on un-flipped diffs removes the concept (the mean). Fixed.
  - Probe split leaked pairs across train/test. Fixed (grouped split).
  - Streaming repetition penalty had the wrong sign for negative logits. Fixed.

Based on: "Representation Engineering: A Top-Down Approach to AI Transparency"
          — Zou et al. (2023)
"""

import os
import json
import torch
import numpy as np
from typing import Optional, Dict, List, Callable
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import vector_math as vm
from config import (
    MODEL_NAME, DEVICE, DTYPE, TARGET_LAYER,
    N_PCA_COMPONENTS, TOKEN_POSITION, VECTORS_DIR,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY,
    DEFAULT_STEERING_COEFF, EXTRACTION_METHOD, RANDOM_SEED,
)


# ══════════════════════════════════════════════════════════════════
#  Hooks
# ══════════════════════════════════════════════════════════════════
class ActivationHook:
    """Forward hook that captures a layer's output hidden states (as float32 on CPU)."""

    def __init__(self):
        self.activation: Optional[torch.Tensor] = None
        self._handle = None

    def hook_fn(self, module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        self.activation = hs.detach().float().cpu()

    def register(self, layer_module):
        self._handle = layer_module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def get_activation(self, token_position: int = -1) -> Optional[torch.Tensor]:
        """Activation at one token position, shape (hidden_dim,)."""
        if self.activation is None:
            return None
        return self.activation[0, token_position, :]

    def get_full_activation(self) -> Optional[torch.Tensor]:
        """All tokens, shape (seq_len, hidden_dim)."""
        if self.activation is None:
            return None
        return self.activation[0]


class SteeringHook:
    """Forward hook that ADDS `coefficient * steering_vector` to a layer's hidden states."""

    def __init__(self, steering_vector: torch.Tensor, coefficient: float = 1.0):
        self.steering_vector = steering_vector
        self.coefficient = coefficient
        self._handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
            delta = self.coefficient * self.steering_vector.to(hs.device, dtype=hs.dtype)
            return (hs + delta.unsqueeze(0).unsqueeze(0),) + output[1:]
        delta = self.coefficient * self.steering_vector.to(output.device, dtype=output.dtype)
        return output + delta.unsqueeze(0).unsqueeze(0)

    def register(self, layer_module):
        self._handle = layer_module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ══════════════════════════════════════════════════════════════════
#  Engine
# ══════════════════════════════════════════════════════════════════
class RepEEngine:
    def __init__(self, model_name: str = MODEL_NAME, target_layer: int = TARGET_LAYER,
                 method: str = EXTRACTION_METHOD, seed: int = RANDOM_SEED):
        self.model_name = model_name
        self.target_layer = target_layer
        self.method = method
        self.seed = seed
        self.model = None
        self.tokenizer = None

        # Per-concept storage
        self.reading_vectors: Dict[str, torch.Tensor] = {}       # unit-norm direction at target_layer
        self.steer_scales: Dict[str, float] = {}                 # activation units per coefficient=1
        self.calibrations: Dict[str, Dict[str, float]] = {}      # pos/neg mean projections
        self.methods: Dict[str, str] = {}
        self.extraction_info: Dict[str, Dict] = {}
        self.pca_models: Dict[str, PCA] = {}
        self.pca_explained_variances: Dict[str, np.ndarray] = {}
        self.all_difference_vectors: Dict[str, np.ndarray] = {}  # at target_layer
        self.pair_acts_all: Dict[str, tuple] = {}                # (pos, neg) each (n_pairs, n_layers, dim)
        self.layer_vectors: Dict[str, Dict[int, Dict]] = {}      # concept -> layer -> {"vector","scale"}

        self.active_concept: str = "honesty"

        self.probes: Dict[str, object] = {}
        self.probe_metrics: Dict[str, Dict] = {}

        self._steering_hooks: List[SteeringHook] = []
        self._model_loaded = False

    # ── Backward-compat properties (v2 API) ───────────────────────
    @property
    def reading_vector(self) -> Optional[torch.Tensor]:
        return self.reading_vectors.get(self.active_concept)

    @reading_vector.setter
    def reading_vector(self, value):
        if value is not None:
            self.reading_vectors[self.active_concept] = value
        elif self.active_concept in self.reading_vectors:
            del self.reading_vectors[self.active_concept]

    @property
    def pca_model(self) -> Optional[PCA]:
        return self.pca_models.get(self.active_concept)

    @pca_model.setter
    def pca_model(self, value):
        if value is not None:
            self.pca_models[self.active_concept] = value

    @property
    def pca_explained_variance(self) -> Optional[np.ndarray]:
        return self.pca_explained_variances.get(self.active_concept)

    @pca_explained_variance.setter
    def pca_explained_variance(self, value):
        if value is not None:
            self.pca_explained_variances[self.active_concept] = value

    # ── Model plumbing ────────────────────────────────────────────
    def load_model(self):
        if self._model_loaded:
            return
        print(f"🔄 Loading model: {self.model_name}")
        print(f"   Device: {DEVICE} | Dtype: {DTYPE}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True,
        )
        self.model.eval()
        self._model_loaded = True
        print("✅ Model loaded successfully!")
        print(f"   Layers: {len(self._get_layers())} | Hidden dim: {self.model.config.hidden_size}"
              f" | Target layer: {self.target_layer}")

    def _get_layers(self):
        if hasattr(self.model, "model"):
            if hasattr(self.model.model, "layers"):
                return self.model.model.layers
            if hasattr(self.model.model, "decoder"):
                return self.model.model.decoder.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        raise ValueError(f"Cannot find transformer layers for model: {self.model_name}")

    def _get_target_layer_module(self, layer_idx: Optional[int] = None):
        layers = self._get_layers()
        idx = layer_idx if layer_idx is not None else self.target_layer
        if not 0 <= idx < len(layers):
            raise ValueError(f"Layer {idx} out of range; model has layers 0..{len(layers) - 1}.")
        return layers[idx]

    def _tokenize(self, text: str, max_length: int = 512):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        return {k: v.to(DEVICE) for k, v in inputs.items()}

    # ══════════════════════════════════════════════════════════════
    #  Activation extraction
    # ══════════════════════════════════════════════════════════════
    @torch.no_grad()
    def extract_activation(self, text: str, token_position: int = TOKEN_POSITION,
                           layer: Optional[int] = None) -> torch.Tensor:
        """Hidden state at one layer & token position, shape (hidden_dim,), float32 CPU."""
        self.load_model()
        self.unsteer()  # never read activations while a steering hook is active
        hook = ActivationHook().register(self._get_target_layer_module(layer))
        try:
            self.model(**self._tokenize(text))
            act = hook.get_activation(token_position)
            if act is None:
                raise RuntimeError("Failed to capture activation — hook did not fire.")
            return act
        finally:
            hook.remove()

    @torch.no_grad()
    def extract_activation_all_layers(self, text: str, token_position: int = TOKEN_POSITION) -> np.ndarray:
        """One forward pass -> (n_layers, hidden_dim) float32 array."""
        self.load_model()
        self.unsteer()
        hooks = [ActivationHook().register(l) for l in self._get_layers()]
        try:
            self.model(**self._tokenize(text))
            return np.stack([h.get_activation(token_position).numpy() for h in hooks]).astype(np.float32)
        finally:
            for h in hooks:
                h.remove()

    def extract_pair_activations(
        self,
        prompt_pairs: List[Dict[str, str]],
        concept_name: str = "honesty",
        token_position: int = TOKEN_POSITION,
        show_progress: bool = True,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """
        Run every pair through the model once (all layers captured) and cache:
          pair_acts_all[concept] = (pos, neg), each (n_pairs, n_layers, dim)
        Returns the target-layer difference vectors (n_pairs, dim).
        """
        self.load_model()
        pos_all, neg_all = [], []
        it = tqdm(prompt_pairs, desc=f"🧠 Extracting {concept_name}") if show_progress else prompt_pairs
        for i, pair in enumerate(it):
            pos_all.append(self.extract_activation_all_layers(pair["positive"], token_position))
            neg_all.append(self.extract_activation_all_layers(pair["negative"], token_position))
            if progress_cb:
                progress_cb(i + 1, len(prompt_pairs))
        pos, neg = np.stack(pos_all), np.stack(neg_all)
        self.pair_acts_all[concept_name] = (pos, neg)
        diffs = (pos[:, self.target_layer] - neg[:, self.target_layer]).astype(np.float32)
        self.all_difference_vectors[concept_name] = diffs
        print(f"✅ Extracted {len(prompt_pairs)} pairs × {pos.shape[1]} layers for '{concept_name}' "
              f"(diffs at layer {self.target_layer}: {diffs.shape})")
        return diffs

    def extract_contrastive_vectors(self, prompt_pairs, token_position: int = TOKEN_POSITION,
                                    show_progress: bool = True, concept_name: str = "honesty") -> np.ndarray:
        """v2-compatible wrapper around extract_pair_activations."""
        return self.extract_pair_activations(prompt_pairs, concept_name, token_position, show_progress)

    # ══════════════════════════════════════════════════════════════
    #  Reading vector
    # ══════════════════════════════════════════════════════════════
    def compute_reading_vector(
        self,
        difference_vectors: Optional[np.ndarray] = None,
        n_components: int = N_PCA_COMPONENTS,
        concept_name: str = "honesty",
        method: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Turn the difference vectors into a unit-norm direction (+ steer scale,
        calibration, and — if all-layer activations are cached — one direction per layer).
        """
        if difference_vectors is None:
            difference_vectors = self.all_difference_vectors.get(concept_name)
        if difference_vectors is None:
            raise ValueError("No difference vectors available. Run extract_pair_activations first.")
        method = method or self.method

        print(f"📊 Computing '{concept_name}' direction: method={method}, n_pairs={len(difference_vectors)}")
        d, info = vm.compute_direction(difference_vectors, method, n_components, seed=self.seed)
        vec = torch.tensor(d, dtype=torch.float32)

        self.reading_vectors[concept_name] = vec
        self.methods[concept_name] = method
        self.steer_scales[concept_name] = vm.steer_scale(difference_vectors, d)
        self.pca_explained_variances[concept_name] = np.array(info["explained_variance_ratio"])
        if "pca" in info:
            self.pca_models[concept_name] = info["pca"]
        self.extraction_info[concept_name] = {
            "cos_with_mean_diff": info["cos_with_mean_diff"], "method": method,
        }

        if concept_name in self.pair_acts_all:
            pos, neg = self.pair_acts_all[concept_name]
            self.calibrations[concept_name] = vm.calibration(pos[:, self.target_layer] @ d,
                                                             neg[:, self.target_layer] @ d)
            self._compute_layer_vectors(concept_name, method, n_components)

        print(f"✅ '{concept_name}' vector ready | PC1/dir energy: "
              f"{self.pca_explained_variances[concept_name][0] * 100:.1f}% | "
              f"cos(dir, mean-diff)={info['cos_with_mean_diff']:.3f} | "
              f"steer scale={self.steer_scales[concept_name]:.2f}")
        return vec

    def _compute_layer_vectors(self, concept_name: str, method: str, n_components: int = N_PCA_COMPONENTS):
        pos, neg = self.pair_acts_all[concept_name]
        out = {}
        for l in range(pos.shape[1]):
            diffs = pos[:, l] - neg[:, l]
            d, _ = vm.compute_direction(diffs, method, n_components, seed=self.seed)
            out[l] = {"vector": torch.tensor(d, dtype=torch.float32),
                      "scale": vm.steer_scale(diffs, d)}
        self.layer_vectors[concept_name] = out

    def _ensure_correct_sign(self, reading_vector: torch.Tensor, prompt_pairs=None) -> torch.Tensor:
        """
        v2 compat. Sign is now aligned with the mean difference vector inside
        compute_direction(), so this is a no-op (and costs no extra forward passes).
        """
        return reading_vector

    # ── Persistence ───────────────────────────────────────────────
    def save_vector(self, name: str = "honesty", metadata: Optional[Dict] = None):
        vec = self.reading_vectors.get(name)
        if vec is None:
            raise ValueError(f"No reading vector for '{name}'. Run compute_reading_vector first.")

        torch.save(vec, os.path.join(VECTORS_DIR, f"{name}_vector.pt"))
        meta_path = os.path.join(VECTORS_DIR, f"{name}_metadata.json")

        if name in self.layer_vectors:
            torch.save({int(l): {"vector": v["vector"], "scale": float(v["scale"])}
                        for l, v in self.layer_vectors[name].items()},
                       os.path.join(VECTORS_DIR, f"{name}_layers.pt"))

        meta = {
            "name": name,
            "model": self.model_name,
            "target_layer": self.target_layer,
            "hidden_dim": int(vec.shape[0]),
            "vector_norm": float(torch.norm(vec).item()),
            "method": self.methods.get(name, self.method),
            "steer_scale": self.steer_scales.get(name),
            "calibration": self.calibrations.get(name),
            "cos_with_mean_diff": self.extraction_info.get(name, {}).get("cos_with_mean_diff"),
            "has_layer_vectors": name in self.layer_vectors,
            "pca_explained_variance": self.pca_explained_variances[name].tolist()
            if name in self.pca_explained_variances else None,
            "format_version": 3,
        }
        if metadata:
            meta.update(metadata)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"💾 Saved '{name}' (vector + metadata{' + per-layer vectors' if name in self.layer_vectors else ''}) → {VECTORS_DIR}")

    def load_vector(self, name: str = "honesty") -> torch.Tensor:
        vector_path = os.path.join(VECTORS_DIR, f"{name}_vector.pt")
        if not os.path.exists(vector_path):
            raise FileNotFoundError(f"No saved vector found at: {vector_path}")

        vec = torch.load(vector_path, map_location="cpu", weights_only=True)
        self.reading_vectors[name] = vec
        self.active_concept = name

        meta_path = os.path.join(VECTORS_DIR, f"{name}_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("target_layer") is not None and meta["target_layer"] != self.target_layer:
                print(f"⚠️  '{name}' was extracted at layer {meta['target_layer']}; "
                      f"switching target layer {self.target_layer} → {meta['target_layer']}.")
                self.target_layer = meta["target_layer"]
            self.methods[name] = meta.get("method", "pca_raw")
            if meta.get("steer_scale") is not None:
                self.steer_scales[name] = float(meta["steer_scale"])
            else:
                print(f"⚠️  '{name}' is a v2 vector without a steer scale — coefficients act on a bare "
                      f"unit vector and will be very weak. Re-run extract_vectors.py to upgrade it.")
                self.steer_scales[name] = 1.0
            if meta.get("calibration"):
                self.calibrations[name] = meta["calibration"]

        layers_path = os.path.join(VECTORS_DIR, f"{name}_layers.pt")
        if os.path.exists(layers_path):
            raw = torch.load(layers_path, map_location="cpu", weights_only=True)
            self.layer_vectors[name] = {int(l): v for l, v in raw.items()}

        print(f"📂 Loaded '{name}' | norm={torch.norm(vec).item():.3f} | layer={self.target_layer} | "
              f"scale={self.steer_scales.get(name, 1.0):.2f}")
        return vec

    # ══════════════════════════════════════════════════════════════
    #  Steering
    # ══════════════════════════════════════════════════════════════
    def steer(self, coefficient: float = DEFAULT_STEERING_COEFF, concept_name: Optional[str] = None):
        """
        Add `coefficient × steer_scale × direction` to the residual stream at the target layer.
        coefficient is in units of the concept gap: +1 ≈ push one typical negative→positive step.
        """
        self.load_model()
        self.unsteer()
        cname = concept_name or self.active_concept
        vec = self.reading_vectors.get(cname)
        if vec is None:
            raise ValueError(f"No reading vector available for '{cname}'. Extract or load one first.")
        scale = self.steer_scales.get(cname, 1.0)
        hook = SteeringHook(vec * scale, coefficient).register(self._get_target_layer_module())
        self._steering_hooks.append(hook)
        print(f"🎯 Steering: concept={cname}, c={coefficient:+.2f} (×{scale:.2f} activation units)")

    def steer_vector(self, unit_vector: torch.Tensor, coefficient: float, scale: float = 1.0,
                     layer: Optional[int] = None):
        """Steer with an arbitrary direction (used for random-vector control experiments)."""
        self.load_model()
        self.unsteer()
        hook = SteeringHook(unit_vector * scale, coefficient).register(self._get_target_layer_module(layer))
        self._steering_hooks.append(hook)

    def steer_multi_layer(self, layer_coefficients: Dict[int, float], concept_name: Optional[str] = None):
        """
        Inject at several layers at once. Uses each layer's OWN direction & scale when
        per-layer vectors are available (a direction found at layer 8 isn't the "same"
        direction at layer 12); falls back to the target-layer vector otherwise.
        """
        self.load_model()
        self.unsteer()
        cname = concept_name or self.active_concept
        base = self.reading_vectors.get(cname)
        if base is None:
            raise ValueError(f"No reading vector for '{cname}'.")
        per_layer = self.layer_vectors.get(cname, {})

        for layer_idx, coeff in layer_coefficients.items():
            layer_idx = int(layer_idx)
            if coeff == 0.0:
                continue
            if layer_idx in per_layer:
                v, s = per_layer[layer_idx]["vector"], per_layer[layer_idx]["scale"]
            else:
                v, s = base, self.steer_scales.get(cname, 1.0)
            self._steering_hooks.append(SteeringHook(v * s, coeff).register(self._get_target_layer_module(layer_idx)))
        active = {k: v for k, v in layer_coefficients.items() if v != 0.0}
        print(f"🎯 Multi-layer steering: {active} ({'per-layer' if per_layer else 'shared'} vectors)")

    def unsteer(self):
        for hook in self._steering_hooks:
            hook.remove()
        self._steering_hooks = []

    # ══════════════════════════════════════════════════════════════
    #  Generation
    # ══════════════════════════════════════════════════════════════
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS,
                 temperature: float = TEMPERATURE, top_p: float = TOP_P,
                 repetition_penalty: float = REPETITION_PENALTY, seed: Optional[int] = None) -> str:
        """Completion only (no prompt). Pass `seed` for reproducible comparisons across coefficients."""
        self.load_model()
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self._tokenize(prompt)
        kwargs = dict(max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty,
                      pad_token_id=self.tokenizer.pad_token_id)
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kwargs.update(do_sample=False)
        out = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def generate_streaming(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS,
                           temperature: float = TEMPERATURE, top_p: float = TOP_P,
                           repetition_penalty: float = REPETITION_PENALTY, seed: Optional[int] = None):
        """Yield decoded tokens one at a time."""
        self.load_model()
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self._tokenize(prompt)
        n_prompt = inputs["input_ids"].shape[1]
        past, generated = None, []

        for step in range(max_new_tokens):
            if step == 0:
                model_inputs = inputs
            else:
                model_inputs = {
                    "input_ids": torch.tensor([[generated[-1]]], device=DEVICE),
                    "attention_mask": torch.ones(1, n_prompt + len(generated), device=DEVICE),
                    "past_key_values": past,
                }
            outputs = self.model(**model_inputs, use_cache=True)
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :].float()

            # HF-style repetition penalty: shrink positive logits, push negative ones further down
            if repetition_penalty != 1.0 and generated:
                idx = torch.tensor(sorted(set(generated)), device=logits.device)
                sel = logits[0, idx]
                logits[0, idx] = torch.where(sel < 0, sel * repetition_penalty, sel / repetition_penalty)

            if not temperature or temperature <= 0:
                token_id = int(torch.argmax(logits, dim=-1).item())
            else:
                logits = logits / temperature
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                sorted_logits[cum - probs > top_p] = float("-inf")
                probs = torch.softmax(sorted_logits, dim=-1)
                token_id = int(sorted_idx.gather(1, torch.multinomial(probs, 1)).item())

            if token_id == self.tokenizer.eos_token_id:
                break
            generated.append(token_id)
            yield self.tokenizer.decode([token_id], skip_special_tokens=True)

    def compare_steered_outputs(self, prompt: str, coefficients: Optional[List[float]] = None,
                                max_new_tokens: int = MAX_NEW_TOKENS, concept_name: Optional[str] = None,
                                seed: Optional[int] = RANDOM_SEED) -> List[Dict]:
        """Generate at several coefficients using the SAME sampling seed so differences come from steering."""
        from config import COMPARE_COEFFICIENTS
        coefficients = coefficients if coefficients is not None else COMPARE_COEFFICIENTS
        cname = concept_name or self.active_concept
        results = []
        for coeff in coefficients:
            if coeff == 0.0:
                self.unsteer()
                label = "baseline (no steering)"
            else:
                self.steer(coeff, concept_name=cname)
                label = f"coefficient = {coeff:+.2f}"
            print(f"\n{'=' * 60}\n  Generating with {label}\n{'=' * 60}")
            output = self.generate(prompt, max_new_tokens=max_new_tokens, seed=seed)
            results.append({"coefficient": coeff, "label": label, "output": output})
            print(output)
        self.unsteer()
        return results

    # ══════════════════════════════════════════════════════════════
    #  Reading ("mind reading")
    # ══════════════════════════════════════════════════════════════
    def project_onto_vector(self, text: str, concept_name: Optional[str] = None) -> float:
        """Raw scalar projection of the last-token activation onto the unit reading vector."""
        cname = concept_name or self.active_concept
        vec = self.reading_vectors.get(cname)
        if vec is None:
            raise ValueError(f"No reading vector for '{cname}'.")
        return float(torch.dot(self.extract_activation(text), vec).item())

    def read_concept(self, text: str, concept_name: Optional[str] = None) -> Dict:
        """
        Calibrated reading: `normalized` is 0 at the typical NEGATIVE training example and
        1 at the typical POSITIVE one (can go outside [0,1]).
        """
        cname = concept_name or self.active_concept
        proj = self.project_onto_vector(text, cname)
        cal = self.calibrations.get(cname)
        norm = vm.normalized_score(proj, cal)
        return {
            "concept": cname, "projection": proj, "normalized": norm,
            "label": None if cal is None else ("positive" if proj > cal["threshold"] else "negative"),
        }

    @torch.no_grad()
    def scan_tokens(self, text: str, concept_name: Optional[str] = None, layer: Optional[int] = None) -> Dict:
        """
        Per-token projection onto the concept direction (the RepE "lie detector" view).

        `relative` = projection minus the text's mean projection (BOS excluded: attention-sink
        tokens have massive activations that would swamp everything). Colour by `relative`.
        """
        self.load_model()
        self.unsteer()
        cname = concept_name or self.active_concept
        vec = self.reading_vectors.get(cname)
        if vec is None:
            raise ValueError(f"No reading vector for '{cname}'.")
        if layer is not None and cname in self.layer_vectors and layer in self.layer_vectors[cname]:
            vec = self.layer_vectors[cname][layer]["vector"]

        hook = ActivationHook().register(self._get_target_layer_module(layer))
        try:
            inputs = self._tokenize(text, max_length=256)
            self.model(**inputs)
            acts = hook.get_full_activation()                     # (seq, dim)
        finally:
            hook.remove()

        ids = inputs["input_ids"][0].tolist()
        tokens = [self.tokenizer.decode([i]) for i in ids]
        proj = (acts @ vec).numpy()
        skip_first = self.tokenizer.bos_token_id is not None and ids[0] == self.tokenizer.bos_token_id
        body = proj[1:] if skip_first else proj
        mean = float(body.mean()) if len(body) else 0.0
        relative = proj - mean
        if skip_first:
            relative[0] = 0.0
        cal = self.calibrations.get(cname)
        return {
            "concept": cname, "layer": layer if layer is not None else self.target_layer,
            "tokens": tokens, "projection": proj.tolist(), "relative": relative.tolist(),
            "mean_projection": mean, "bos_skipped": bool(skip_first),
            "normalized_mean": vm.normalized_score(mean, cal),
        }

    @torch.no_grad()
    def continuation_nll(self, prompt: str, continuation: str) -> Optional[float]:
        """
        Mean negative log-likelihood per token of `continuation` given `prompt` under the
        UNSTEERED model. A fluency/coherence proxy: heavy steering raises it.
        """
        self.load_model()
        self.unsteer()
        n_prompt = len(self.tokenizer(prompt)["input_ids"])
        inputs = self._tokenize(prompt + continuation, max_length=768)
        ids = inputs["input_ids"]
        if ids.shape[1] <= n_prompt:
            return None
        logits = self.model(**inputs).logits[0, :-1].float()
        logp = torch.log_softmax(logits, dim=-1)
        tgt = ids[0, 1:]
        nll = -logp[torch.arange(len(tgt), device=tgt.device), tgt][n_prompt - 1:]
        return float(nll.mean().item())

    # ══════════════════════════════════════════════════════════════
    #  Heatmap (all layers × tokens)
    # ══════════════════════════════════════════════════════════════
    @torch.no_grad()
    def extract_all_layer_activations(self, text: str) -> Dict:
        self.load_model()
        self.unsteer()
        layers = self._get_layers()
        n_layers = len(layers)
        hooks = [ActivationHook().register(l) for l in layers]
        try:
            inputs = self._tokenize(text, max_length=256)
            self.model(**inputs)
            ids = inputs["input_ids"][0].tolist()
            tokens = [self.tokenizer.decode([i]) for i in ids]
            n_tokens = len(tokens)

            norms = np.zeros((n_layers, n_tokens), dtype=np.float32)
            proj = {c: np.zeros((n_layers, n_tokens), dtype=np.float32) for c in self.reading_vectors}
            for li, hook in enumerate(hooks):
                act = hook.get_full_activation()
                if act is None:
                    continue
                norms[li] = torch.norm(act, dim=-1).numpy()
                for cname, vec in self.reading_vectors.items():
                    # use each layer's own direction when we have one
                    v = self.layer_vectors.get(cname, {}).get(li, {}).get("vector", vec)
                    proj[cname][li] = (act @ v).numpy()
            return {
                "tokens": tokens, "n_layers": n_layers, "n_tokens": n_tokens,
                "activation_norms": norms.tolist(),
                "concept_projections": {k: v.tolist() for k, v in proj.items()},
                "note": "Token 0 (BOS) has a massive-activation outlier; ignore it when reading colours.",
            }
        finally:
            for h in hooks:
                h.remove()

    # ══════════════════════════════════════════════════════════════
    #  Probes (grouped split — no pair leakage)
    # ══════════════════════════════════════════════════════════════
    def _pair_acts_at_target(self, prompt_pairs, concept_name):
        cached = self.pair_acts_all.get(concept_name)
        if cached is None or len(cached[0]) != len(prompt_pairs):
            self.extract_pair_activations(prompt_pairs, concept_name)
        pos, neg = self.pair_acts_all[concept_name]
        return pos[:, self.target_layer], neg[:, self.target_layer]

    def train_probe(self, prompt_pairs: List[Dict[str, str]], concept_name: str = "honesty",
                    test_size: float = 0.2) -> Dict:
        """
        Logistic-regression probe on target-layer activations. Train/test split is by
        PAIR, so a held-out prompt's twin is never in the training set.
        """
        self.load_model()
        print(f"🔬 Training probe for '{concept_name}' on {len(prompt_pairs)} pairs...")
        pos, neg = self._pair_acts_at_target(prompt_pairs, concept_name)
        tr, te = vm.grouped_split(len(pos), test_size, seed=self.seed)

        X_tr = np.vstack([pos[tr], neg[tr]]); y_tr = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
        X_te = np.vstack([pos[te], neg[te]]); y_te = np.r_[np.ones(len(te)), np.zeros(len(te))]

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        # How aligned is the supervised probe with the unsupervised reading vector?
        w = clf[-1].coef_[0] / clf[0].scale_
        cos = None
        if concept_name in self.reading_vectors:
            cos = vm.cosine(w, self.reading_vectors[concept_name].numpy())

        metrics = {
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "precision": float(precision_score(y_te, y_pred, zero_division=0)),
            "recall": float(recall_score(y_te, y_pred, zero_division=0)),
            "f1": float(f1_score(y_te, y_pred, zero_division=0)),
            "auroc": float(roc_auc_score(y_te, y_prob)),
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "cosine_with_reading_vector": cos,
            "split": "grouped-by-pair",
        }
        self.probes[concept_name] = clf
        self.probe_metrics[concept_name] = metrics
        print(f"✅ Probe '{concept_name}': acc={metrics['accuracy']:.3f} auroc={metrics['auroc']:.3f} "
              f"f1={metrics['f1']:.3f} cos(probe, reading vec)={cos if cos is None else round(cos, 3)}")
        return metrics

    def probe_text(self, text: str, concept_name: str = "honesty") -> Dict:
        if concept_name not in self.probes:
            raise ValueError(f"No trained probe for '{concept_name}'. Run train_probe first.")
        clf = self.probes[concept_name]
        act = self.extract_activation(text).numpy().reshape(1, -1)
        proba = clf.predict_proba(act)[0]
        return {"predicted_label": int(clf.predict(act)[0]),
                "probability_positive": float(proba[1]), "probability_negative": float(proba[0])}

    # ══════════════════════════════════════════════════════════════
    #  Layer sweep (held-out) — replaces the old norm-of-difference scan
    # ══════════════════════════════════════════════════════════════
    def layer_sweep(self, prompt_pairs: List[Dict[str, str]], concept_name: str = "honesty",
                    test_size: float = 0.3, method: Optional[str] = None) -> Dict:
        """
        For every layer: fit a direction on TRAIN pairs, then measure how well it separates
        positive from negative on held-out TEST pairs (pair accuracy, AUROC, d').
        """
        self.load_model()
        method = method or self.method
        cached = self.pair_acts_all.get(concept_name)
        if cached is None or len(cached[0]) != len(prompt_pairs):
            self.extract_pair_activations(prompt_pairs, concept_name)
        pos, neg = self.pair_acts_all[concept_name]
        n_pairs, n_layers = pos.shape[0], pos.shape[1]
        tr, te = vm.grouped_split(n_pairs, test_size, seed=self.seed)

        rows = []
        for l in range(n_layers):
            d, _ = vm.compute_direction(pos[tr, l] - neg[tr, l], method, N_PCA_COMPONENTS, seed=self.seed)
            rows.append(vm.pair_metrics(pos[te, l], neg[te, l], d))

        auroc = [r["auroc"] for r in rows]
        dprime = [r["d_prime"] for r in rows]
        best = int(max(range(n_layers), key=lambda i: (round(auroc[i], 4), dprime[i])))
        return {
            "layer_indices": list(range(n_layers)),
            "separability_scores": dprime,          # v2 key kept for the dashboard
            "auroc": auroc,
            "pair_accuracy": [r["pair_accuracy"] for r in rows],
            "d_prime": dprime,
            "best_layer": best,
            "n_train_pairs": int(len(tr)), "n_test_pairs": int(len(te)), "method": method,
        }

    def get_layer_scan_data(self, prompt_pairs, n_pairs: int = 40, concept_name: str = "honesty") -> Dict:
        """v2-compatible entry point (now a held-out sweep rather than diff-norm)."""
        return self.layer_sweep(prompt_pairs[:n_pairs], concept_name)

    # ══════════════════════════════════════════════════════════════
    #  Visualisation helpers
    # ══════════════════════════════════════════════════════════════
    def get_pca_projection_data(self, concept_name: Optional[str] = None) -> Optional[Dict]:
        cname = concept_name or self.active_concept
        pca, diffs = self.pca_models.get(cname), self.all_difference_vectors.get(cname)
        if pca is None or diffs is None or pca.n_components_ < 2:
            return None
        p = pca.transform(diffs)
        return {"pc1": p[:, 0].tolist(), "pc2": p[:, 1].tolist(),
                "explained_variance": pca.explained_variance_ratio_.tolist()}

    def get_activation_scatter(self, concept_name: Optional[str] = None) -> Optional[Dict]:
        """2-D PCA of the raw positive+negative activations: do the two classes separate?"""
        cname = concept_name or self.active_concept
        cached = self.pair_acts_all.get(cname)
        if cached is None:
            return None
        pos, neg = cached[0][:, self.target_layer], cached[1][:, self.target_layer]
        pca = PCA(n_components=2, random_state=self.seed).fit(np.vstack([pos, neg]))
        return {"positive": pca.transform(pos).tolist(), "negative": pca.transform(neg).tolist(),
                "explained_variance": pca.explained_variance_ratio_.tolist(), "layer": self.target_layer}

    def concept_similarity(self) -> Dict:
        """Cosine similarity between all loaded concept directions (are 'honesty' and 'sycophancy' the same axis?)."""
        names = list(self.reading_vectors.keys())
        M = [[vm.cosine(self.reading_vectors[a].numpy(), self.reading_vectors[b].numpy()) for b in names]
             for a in names]
        return {"concepts": names, "matrix": M}

    # ══════════════════════════════════════════════════════════════
    #  Status
    # ══════════════════════════════════════════════════════════════
    def get_status(self) -> Dict:
        active = self.reading_vectors.get(self.active_concept)
        return {
            "model_loaded": self._model_loaded,
            "model_name": self.model_name,
            "target_layer": self.target_layer,
            "n_layers": len(self._get_layers()) if self._model_loaded else None,
            "device": DEVICE,
            "active_concept": self.active_concept,
            "available_concepts": list(self.reading_vectors.keys()),
            "has_reading_vector": active is not None,
            "vector_norm": float(torch.norm(active).item()) if active is not None else None,
            "steer_scale": self.steer_scales.get(self.active_concept),
            "method": self.methods.get(self.active_concept),
            "calibration": self.calibrations.get(self.active_concept),
            "has_layer_vectors": self.active_concept in self.layer_vectors,
            "is_steering": len(self._steering_hooks) > 0,
            "steering_coefficient": self._steering_hooks[0].coefficient if self._steering_hooks else None,
            "pca_variance_explained": self.pca_explained_variances[self.active_concept].tolist()
            if self.active_concept in self.pca_explained_variances else None,
            "available_probes": list(self.probes.keys()),
            "probe_metrics": self.probe_metrics,
        }

    def get_available_saved_vectors(self) -> List[Dict]:
        out = []
        if os.path.exists(VECTORS_DIR):
            for f in sorted(os.listdir(VECTORS_DIR)):
                if f.endswith("_metadata.json"):
                    with open(os.path.join(VECTORS_DIR, f)) as fh:
                        out.append(json.load(fh))
        return out
