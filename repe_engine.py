"""
repe_engine.py — Core Representation Engineering Engine (v2.0)

This module implements the full RepE pipeline:
  1. Activation extraction via PyTorch forward hooks
  2. Contrastive difference computation
  3. PCA-based vector isolation (the "Reading Vector")
  4. Real-time steering via activation injection
  5. Multi-layer activation heatmaps
  6. Linear probe classifier for concept detection
  7. Multi-layer simultaneous steering

Based on: "Representation Engineering: A Top-Down Approach to AI Transparency"
           — Zou, Phan, Chen, Campbell, Guo, Ren, Pan, Yin, Mazeika,
             Dombrowski, Jha, Hendrycks (2023)
"""

import os
import json
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    MODEL_NAME, DEVICE, DTYPE, TARGET_LAYER,
    N_PCA_COMPONENTS, TOKEN_POSITION, VECTORS_DIR,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY,
    DEFAULT_STEERING_COEFF,
)


class ActivationHook:
    """
    A reusable forward hook that captures hidden-state activations
    from a specific transformer layer during the forward pass.
    """

    def __init__(self):
        self.activation: Optional[torch.Tensor] = None
        self._handle = None

    def hook_fn(self, module, input, output):
        """
        Called automatically by PyTorch during forward pass.
        `output` is typically a tuple: (hidden_states, ...).
        We grab the hidden states tensor.
        """
        if isinstance(output, tuple):
            self.activation = output[0].detach().cpu()
        else:
            self.activation = output.detach().cpu()

    def register(self, layer_module):
        """Attach the hook to a transformer layer."""
        self._handle = layer_module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        """Detach the hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def get_activation(self, token_position: int = -1) -> Optional[torch.Tensor]:
        """
        Return the activation at a specific token position.
        Shape: (hidden_dim,)
        """
        if self.activation is None:
            return None
        # activation shape: (batch=1, seq_len, hidden_dim)
        return self.activation[0, token_position, :]

    def get_full_activation(self) -> Optional[torch.Tensor]:
        """
        Return the full activation across all tokens.
        Shape: (seq_len, hidden_dim)
        """
        if self.activation is None:
            return None
        return self.activation[0]  # (seq_len, hidden_dim)


class SteeringHook:
    """
    A forward hook that *modifies* hidden states during the forward pass
    by adding a steering vector scaled by a coefficient.
    """

    def __init__(self, steering_vector: torch.Tensor, coefficient: float = 1.0):
        self.steering_vector = steering_vector
        self.coefficient = coefficient
        self._handle = None

    def hook_fn(self, module, input, output):
        """
        Inject the steering vector into the hidden states.
        We add (coefficient × vector) to ALL token positions.
        """
        if isinstance(output, tuple):
            hidden_states = output[0]
            # Broadcast steering vector across batch and sequence dimensions
            delta = self.coefficient * self.steering_vector.to(hidden_states.device, dtype=hidden_states.dtype)
            modified = hidden_states + delta.unsqueeze(0).unsqueeze(0)
            return (modified,) + output[1:]
        else:
            delta = self.coefficient * self.steering_vector.to(output.device, dtype=output.dtype)
            return output + delta.unsqueeze(0).unsqueeze(0)

    def register(self, layer_module):
        """Attach the steering hook to a transformer layer."""
        self._handle = layer_module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        """Detach the hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class RepEEngine:
    """
    The main Representation Engineering engine (v2.0).

    Provides methods for:
      - Loading the model
      - Extracting activations from contrastive prompts (multi-concept)
      - Computing the reading vector via PCA
      - Steering generation with reading vector(s)
      - Multi-layer activation heatmaps
      - Linear probe concept classifier
      - Multi-layer simultaneous steering
    """

    def __init__(self, model_name: str = MODEL_NAME, target_layer: int = TARGET_LAYER):
        self.model_name = model_name
        self.target_layer = target_layer
        self.model = None
        self.tokenizer = None

        # Per-concept storage: concept_name -> data
        self.reading_vectors: Dict[str, torch.Tensor] = {}
        self.pca_models: Dict[str, PCA] = {}
        self.pca_explained_variances: Dict[str, np.ndarray] = {}
        self.all_difference_vectors: Dict[str, np.ndarray] = {}

        # Active concept for backward compat
        self.active_concept: str = "honesty"

        # Probe classifiers: concept_name -> trained model
        self.probes: Dict[str, LogisticRegression] = {}
        self.probe_metrics: Dict[str, Dict] = {}

        # Multi-layer steering hooks
        self._steering_hooks: List[SteeringHook] = []
        self._model_loaded = False

    # ── Backward-compat properties ────────────────────────────────
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

    def load_model(self):
        """Load the model and tokenizer."""
        if self._model_loaded:
            return

        print(f"🔄 Loading model: {self.model_name}")
        print(f"   Device: {DEVICE} | Dtype: {DTYPE}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=DTYPE,
            device_map=DEVICE,
            trust_remote_code=True,
        )
        self.model.eval()
        self._model_loaded = True
        print(f"✅ Model loaded successfully!")
        print(f"   Layers: {len(self._get_layers())}")
        print(f"   Hidden dim: {self.model.config.hidden_size}")
        print(f"   Target layer: {self.target_layer}")

    def _get_layers(self):
        """
        Get the list of transformer layers from the model.
        Handles different model architectures (Llama, Gemma, GPT-2, etc.)
        """
        if hasattr(self.model, 'model'):
            # Llama, Gemma, Mistral, etc.
            if hasattr(self.model.model, 'layers'):
                return self.model.model.layers
            elif hasattr(self.model.model, 'decoder'):
                return self.model.model.decoder.layers
        if hasattr(self.model, 'transformer'):
            # GPT-2, GPT-Neo, etc.
            if hasattr(self.model.transformer, 'h'):
                return self.model.transformer.h
        raise ValueError(f"Cannot find transformer layers for model: {self.model_name}")

    def _get_target_layer_module(self, layer_idx: Optional[int] = None):
        """Get a specific layer module to hook into."""
        layers = self._get_layers()
        idx = layer_idx if layer_idx is not None else self.target_layer
        if idx >= len(layers):
            raise ValueError(
                f"Target layer {idx} exceeds model depth ({len(layers)} layers). "
                f"Try a value between 0 and {len(layers) - 1}."
            )
        return layers[idx]

    @torch.no_grad()
    def extract_activation(self, text: str, token_position: int = TOKEN_POSITION) -> torch.Tensor:
        """
        Run a single text through the model and extract the hidden state
        activation at the target layer.

        Args:
            text: Input text to process
            token_position: Which token's activation to extract (-1 = last)

        Returns:
            Activation tensor of shape (hidden_dim,)
        """
        self.load_model()

        hook = ActivationHook()
        hook.register(self._get_target_layer_module())

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            self.model(**inputs)
            activation = hook.get_activation(token_position)
            if activation is None:
                raise RuntimeError("Failed to capture activation — hook did not fire.")
            return activation
        finally:
            hook.remove()

    def extract_contrastive_vectors(
        self,
        prompt_pairs: List[Dict[str, str]],
        token_position: int = TOKEN_POSITION,
        show_progress: bool = True,
        concept_name: str = "honesty",
    ) -> np.ndarray:
        """
        Phase 1, Step 1: Extract difference vectors from contrastive prompt pairs.

        For each pair:
          difference = activation(positive) - activation(negative)

        Args:
            prompt_pairs: List of dicts with 'positive' and 'negative' keys
            token_position: Token position to extract (-1 = last)
            show_progress: Whether to show a progress bar
            concept_name: Name of the concept being extracted

        Returns:
            numpy array of shape (n_pairs, hidden_dim)
        """
        self.load_model()

        difference_vectors = []
        iterator = tqdm(prompt_pairs, desc=f"🧠 Extracting {concept_name} activations") if show_progress else prompt_pairs

        for pair in iterator:
            pos_activation = self.extract_activation(pair["positive"], token_position)
            neg_activation = self.extract_activation(pair["negative"], token_position)

            # The magic subtraction: isolate the concept direction
            diff = pos_activation - neg_activation
            difference_vectors.append(diff.numpy())

        result = np.array(difference_vectors, dtype=np.float32)
        self.all_difference_vectors[concept_name] = result
        print(f"✅ Extracted {len(difference_vectors)} difference vectors for '{concept_name}'")
        print(f"   Shape: {result.shape}")
        return result

    def compute_reading_vector(
        self,
        difference_vectors: Optional[np.ndarray] = None,
        n_components: int = N_PCA_COMPONENTS,
        concept_name: str = "honesty",
    ) -> torch.Tensor:
        """
        Phase 1, Step 2: Run PCA on the difference vectors to isolate
        the primary direction of the concept.

        The first principal component is the "reading vector" — the direction
        in representation space that maximally captures the concept.

        Args:
            difference_vectors: Array of shape (n_pairs, hidden_dim).
                                Uses previously extracted vectors if None.
            n_components: Number of PCA components to compute.
            concept_name: Name of the concept

        Returns:
            The reading vector as a torch.Tensor of shape (hidden_dim,)
        """
        if difference_vectors is None:
            difference_vectors = self.all_difference_vectors.get(concept_name)
        if difference_vectors is None:
            raise ValueError("No difference vectors available. Run extract_contrastive_vectors first.")

        print(f"📊 Running PCA with {n_components} components on {len(difference_vectors)} vectors...")

        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(difference_vectors)

        # The first component is our reading vector
        reading_vector_np = pca.components_[0]
        reading_vector = torch.tensor(reading_vector_np, dtype=torch.float32)

        # Store in per-concept storage
        self.reading_vectors[concept_name] = reading_vector
        self.pca_models[concept_name] = pca
        self.pca_explained_variances[concept_name] = pca.explained_variance_ratio_

        print(f"✅ Reading vector computed for '{concept_name}'!")
        print(f"   Explained variance ratios: {[f'{v:.4f}' for v in pca.explained_variance_ratio_]}")
        print(f"   PC1 captures {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
        print(f"   Vector norm: {torch.norm(reading_vector).item():.4f}")

        return reading_vector

    def _ensure_correct_sign(self, reading_vector: torch.Tensor, prompt_pairs: List[Dict[str, str]]) -> torch.Tensor:
        """
        PCA doesn't guarantee sign consistency. We check a few prompt pairs
        to ensure that projecting positive prompts onto the vector gives
        a positive dot product.
        """
        self.load_model()

        positive_projections = 0
        n_check = min(10, len(prompt_pairs))

        for pair in prompt_pairs[:n_check]:
            pos_act = self.extract_activation(pair["positive"])
            neg_act = self.extract_activation(pair["negative"])
            diff = pos_act - neg_act

            projection = torch.dot(diff, reading_vector)
            if projection > 0:
                positive_projections += 1

        # If most projections are negative, flip the vector
        if positive_projections < n_check / 2:
            print("🔄 Flipping reading vector sign for consistency")
            return -reading_vector
        return reading_vector

    def save_vector(self, name: str = "honesty", metadata: Optional[Dict] = None):
        """Save the reading vector and metadata to disk."""
        vec = self.reading_vectors.get(name)
        if vec is None:
            raise ValueError(f"No reading vector for '{name}'. Run compute_reading_vector first.")

        vector_path = os.path.join(VECTORS_DIR, f"{name}_vector.pt")
        meta_path = os.path.join(VECTORS_DIR, f"{name}_metadata.json")

        torch.save(vec, vector_path)

        meta = {
            "name": name,
            "model": self.model_name,
            "target_layer": self.target_layer,
            "hidden_dim": int(vec.shape[0]),
            "vector_norm": float(torch.norm(vec).item()),
            "pca_explained_variance": self.pca_explained_variances[name].tolist() if name in self.pca_explained_variances else None,
        }
        if metadata:
            meta.update(metadata)

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"💾 Saved vector to: {vector_path}")
        print(f"💾 Saved metadata to: {meta_path}")

    def load_vector(self, name: str = "honesty") -> torch.Tensor:
        """Load a previously saved reading vector."""
        vector_path = os.path.join(VECTORS_DIR, f"{name}_vector.pt")
        if not os.path.exists(vector_path):
            raise FileNotFoundError(f"No saved vector found at: {vector_path}")

        vec = torch.load(vector_path, map_location="cpu", weights_only=True)
        self.reading_vectors[name] = vec
        self.active_concept = name
        print(f"📂 Loaded vector '{name}' from: {vector_path}")
        print(f"   Shape: {vec.shape}")
        print(f"   Norm: {torch.norm(vec).item():.4f}")
        return vec

    def steer(self, coefficient: float = DEFAULT_STEERING_COEFF, concept_name: Optional[str] = None):
        """
        Phase 2: Attach the steering hook to the model.

        This modifies the model's forward pass so that the reading vector
        is added to the hidden states at the target layer.

        Args:
            coefficient: Scaling factor.
                         Positive → amplify concept (e.g., more honest)
                         Negative → suppress/invert concept (e.g., more deceptive)
                         Zero → no steering (baseline)
            concept_name: Which concept vector to use. Uses active_concept if None.
        """
        self.load_model()
        self.unsteer()  # Remove any existing steering

        cname = concept_name or self.active_concept
        vec = self.reading_vectors.get(cname)
        if vec is None:
            raise ValueError(f"No reading vector available for '{cname}'. Extract or load one first.")

        hook = SteeringHook(vec, coefficient)
        hook.register(self._get_target_layer_module())
        self._steering_hooks.append(hook)
        print(f"🎯 Steering active: concept={cname}, coefficient={coefficient:+.2f}")

    def steer_multi_layer(
        self,
        layer_coefficients: Dict[int, float],
        concept_name: Optional[str] = None,
    ):
        """
        Multi-layer steering: inject the reading vector at multiple layers
        simultaneously, each with its own coefficient.

        Args:
            layer_coefficients: Dict mapping layer_index -> coefficient
            concept_name: Which concept vector to use
        """
        self.load_model()
        self.unsteer()

        cname = concept_name or self.active_concept
        vec = self.reading_vectors.get(cname)
        if vec is None:
            raise ValueError(f"No reading vector for '{cname}'.")

        for layer_idx, coeff in layer_coefficients.items():
            if coeff == 0.0:
                continue
            hook = SteeringHook(vec, coeff)
            hook.register(self._get_target_layer_module(layer_idx))
            self._steering_hooks.append(hook)

        active_layers = {k: v for k, v in layer_coefficients.items() if v != 0.0}
        print(f"🎯 Multi-layer steering active: {active_layers}")

    def unsteer(self):
        """Remove any active steering hooks."""
        for hook in self._steering_hooks:
            hook.remove()
        self._steering_hooks = []

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        repetition_penalty: float = REPETITION_PENALTY,
    ) -> str:
        """
        Generate text from a prompt, optionally steered by the reading vector.

        Args:
            prompt: The input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens

        Returns:
            The generated text (completion only, no prompt)
        """
        self.load_model()

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode only the generated tokens (exclude the prompt)
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        repetition_penalty: float = REPETITION_PENALTY,
    ):
        """
        Generator that yields tokens one at a time for streaming output.
        """
        self.load_model()

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        past_key_values = None
        generated_ids = []

        for step in range(max_new_tokens):
            if step == 0:
                model_inputs = inputs
            else:
                model_inputs = {
                    "input_ids": torch.tensor([[generated_ids[-1]]], device=DEVICE),
                    "attention_mask": torch.ones(1, input_ids.shape[1] + len(generated_ids), device=DEVICE),
                }
                if past_key_values is not None:
                    model_inputs["past_key_values"] = past_key_values

            outputs = self.model(**model_inputs, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            # Apply temperature and top-p
            if temperature > 0:
                logits = logits / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0 and generated_ids:
                for token_id in set(generated_ids):
                    logits[0, token_id] /= repetition_penalty

            # Top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            mask = cumulative_probs - probs > top_p
            sorted_logits[mask] = float('-inf')
            probs = torch.softmax(sorted_logits, dim=-1)
            next_token_sorted = torch.multinomial(probs, 1)
            next_token = sorted_indices.gather(1, next_token_sorted).squeeze()

            token_id = next_token.item()

            # Check for EOS
            if token_id == self.tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            yield token_text

    def compare_steered_outputs(
        self,
        prompt: str,
        coefficients: List[float] = [-5.0, -3.0, 0.0, 3.0, 5.0],
        max_new_tokens: int = MAX_NEW_TOKENS,
        concept_name: Optional[str] = None,
    ) -> List[Dict]:
        """
        Generate outputs at various steering coefficients for comparison.
        """
        cname = concept_name or self.active_concept
        results = []

        for coeff in coefficients:
            if coeff == 0.0:
                self.unsteer()
                label = "baseline (no steering)"
            else:
                self.steer(coeff, concept_name=cname)
                label = f"coefficient = {coeff:+.1f}"

            print(f"\n{'='*60}")
            print(f"  Generating with {label}")
            print(f"{'='*60}")

            output = self.generate(prompt, max_new_tokens=max_new_tokens)
            results.append({
                "coefficient": coeff,
                "label": label,
                "output": output,
            })
            print(output)

        self.unsteer()
        return results

    def project_onto_vector(self, text: str, concept_name: Optional[str] = None) -> float:
        """
        'Read the model's mind' — project a text's activation onto the
        reading vector to measure how much the model's internal state
        aligns with the concept.

        Returns a scalar: positive = aligned with concept, negative = opposed.
        """
        cname = concept_name or self.active_concept
        vec = self.reading_vectors.get(cname)
        if vec is None:
            raise ValueError(f"No reading vector for '{cname}'.")

        activation = self.extract_activation(text)
        normalized_vec = vec / torch.norm(vec)
        projection = torch.dot(activation, normalized_vec).item()
        return projection

    # ══════════════════════════════════════════════════════════════
    #  Neural Activation Heatmap
    # ══════════════════════════════════════════════════════════════

    @torch.no_grad()
    def extract_all_layer_activations(
        self,
        text: str,
    ) -> Dict:
        """
        Extract activations from ALL layers for a given text.
        Returns a heatmap-ready data structure.

        Returns:
            Dict with:
              - tokens: list of token strings
              - activations: shape (n_layers, n_tokens) — L2 norms per token per layer
              - concept_projections: shape (n_layers, n_tokens) — projection onto each concept's vector
        """
        self.load_model()
        layers = self._get_layers()
        n_layers = len(layers)

        # Register hooks on ALL layers
        hooks = []
        for layer in layers:
            hook = ActivationHook()
            hook.register(layer)
            hooks.append(hook)

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs_device = {k: v.to(DEVICE) for k, v in inputs.items()}
            self.model(**inputs_device)

            # Get token strings
            token_ids = inputs["input_ids"][0].tolist()
            tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
            n_tokens = len(tokens)

            # Build activation norm matrix: (n_layers, n_tokens)
            activation_norms = np.zeros((n_layers, n_tokens), dtype=np.float32)
            concept_projections = {}

            for layer_idx, hook in enumerate(hooks):
                full_act = hook.get_full_activation()  # (seq_len, hidden_dim)
                if full_act is not None:
                    # L2 norm per token
                    norms = torch.norm(full_act, dim=-1).numpy()
                    activation_norms[layer_idx, :len(norms)] = norms

                    # Project onto each available concept vector
                    for cname, vec in self.reading_vectors.items():
                        if cname not in concept_projections:
                            concept_projections[cname] = np.zeros((n_layers, n_tokens), dtype=np.float32)
                        normalized_vec = vec / torch.norm(vec)
                        projections = (full_act @ normalized_vec).numpy()
                        concept_projections[cname][layer_idx, :len(projections)] = projections

            return {
                "tokens": tokens,
                "n_layers": n_layers,
                "n_tokens": n_tokens,
                "activation_norms": activation_norms.tolist(),
                "concept_projections": {
                    k: v.tolist() for k, v in concept_projections.items()
                },
            }
        finally:
            for hook in hooks:
                hook.remove()

    # ══════════════════════════════════════════════════════════════
    #  Linear Probe Classifier
    # ══════════════════════════════════════════════════════════════

    def train_probe(
        self,
        prompt_pairs: List[Dict[str, str]],
        concept_name: str = "honesty",
        test_size: float = 0.2,
    ) -> Dict:
        """
        Train a linear probe (logistic regression) classifier on the
        extracted activations to detect the concept.

        Labels:
          1 = positive prompt (e.g., truthful)
          0 = negative prompt (e.g., deceptive)

        Returns metrics dict with accuracy, precision, recall, f1.
        """
        self.load_model()

        print(f"🔬 Training probe for '{concept_name}' on {len(prompt_pairs)} pairs...")

        features = []
        labels = []

        for pair in tqdm(prompt_pairs, desc="Extracting probe features"):
            pos_act = self.extract_activation(pair["positive"]).numpy()
            neg_act = self.extract_activation(pair["negative"]).numpy()
            features.append(pos_act)
            labels.append(1)
            features.append(neg_act)
            labels.append(0)

        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        self.probes[concept_name] = clf
        self.probe_metrics[concept_name] = metrics

        print(f"✅ Probe trained for '{concept_name}':")
        print(f"   Accuracy:  {metrics['accuracy']:.3f}")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall:    {metrics['recall']:.3f}")
        print(f"   F1:        {metrics['f1']:.3f}")

        return metrics

    def probe_text(self, text: str, concept_name: str = "honesty") -> Dict:
        """
        Use the trained probe to classify a text.

        Returns:
            Dict with predicted_label, probability, and raw score.
        """
        if concept_name not in self.probes:
            raise ValueError(f"No trained probe for '{concept_name}'. Run train_probe first.")

        clf = self.probes[concept_name]
        activation = self.extract_activation(text).numpy().reshape(1, -1)
        pred = clf.predict(activation)[0]
        proba = clf.predict_proba(activation)[0]

        return {
            "predicted_label": int(pred),
            "probability_positive": float(proba[1]),
            "probability_negative": float(proba[0]),
        }

    # ══════════════════════════════════════════════════════════════
    #  Layer Scan
    # ══════════════════════════════════════════════════════════════

    def get_layer_scan_data(
        self,
        prompt_pairs: List[Dict[str, str]],
        n_pairs: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Scan all layers to find which one best separates the contrastive concepts.
        Useful for finding the optimal target layer.

        Returns dict with layer indices and their separability scores.
        """
        self.load_model()
        layers = self._get_layers()
        n_pairs = min(n_pairs, len(prompt_pairs))
        scores = []

        for layer_idx in tqdm(range(len(layers)), desc="🔍 Scanning layers"):
            hook = ActivationHook()
            hook.register(layers[layer_idx])

            diffs = []
            for pair in prompt_pairs[:n_pairs]:
                pos_act = self._extract_activation_with_hook(hook, pair["positive"])
                neg_act = self._extract_activation_with_hook(hook, pair["negative"])
                diff_norm = torch.norm(pos_act - neg_act).item()
                diffs.append(diff_norm)

            hook.remove()
            avg_diff = np.mean(diffs)
            scores.append(avg_diff)

        return {
            "layer_indices": list(range(len(layers))),
            "separability_scores": scores,
            "best_layer": int(np.argmax(scores)),
        }

    @torch.no_grad()
    def _extract_activation_with_hook(self, hook: ActivationHook, text: str) -> torch.Tensor:
        """Helper: extract activation using an already-registered hook."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        hook.activation = None
        self.model(**inputs)
        return hook.get_activation(TOKEN_POSITION)

    def get_pca_projection_data(self, concept_name: Optional[str] = None) -> Optional[Dict]:
        """
        Get 2D PCA projection data for visualization.
        Projects all difference vectors onto the first two principal components.
        """
        cname = concept_name or self.active_concept
        pca = self.pca_models.get(cname)
        diff_vecs = self.all_difference_vectors.get(cname)

        if pca is None or diff_vecs is None:
            return None

        projections = pca.transform(diff_vecs)
        return {
            "pc1": projections[:, 0].tolist(),
            "pc2": projections[:, 1].tolist(),
            "explained_variance": pca.explained_variance_ratio_.tolist(),
        }

    def get_status(self) -> Dict:
        """Get the current status of the engine."""
        active_vec = self.reading_vectors.get(self.active_concept)
        return {
            "model_loaded": self._model_loaded,
            "model_name": self.model_name,
            "target_layer": self.target_layer,
            "n_layers": len(self._get_layers()) if self._model_loaded else None,
            "device": DEVICE,
            "active_concept": self.active_concept,
            "available_concepts": list(self.reading_vectors.keys()),
            "has_reading_vector": active_vec is not None,
            "vector_norm": float(torch.norm(active_vec).item()) if active_vec is not None else None,
            "is_steering": len(self._steering_hooks) > 0,
            "steering_coefficient": self._steering_hooks[0].coefficient if self._steering_hooks else None,
            "pca_variance_explained": self.pca_explained_variances.get(self.active_concept, np.array([])).tolist() if self.active_concept in self.pca_explained_variances else None,
            "available_probes": list(self.probes.keys()),
            "probe_metrics": self.probe_metrics,
        }

    def get_available_saved_vectors(self) -> List[Dict]:
        """List all saved vectors on disk."""
        vectors = []
        if os.path.exists(VECTORS_DIR):
            for f in sorted(os.listdir(VECTORS_DIR)):
                if f.endswith("_metadata.json"):
                    with open(os.path.join(VECTORS_DIR, f)) as fh:
                        meta = json.load(fh)
                        vectors.append(meta)
        return vectors
