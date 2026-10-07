"""
repe_engine.py — Core Representation Engineering Engine

This module implements the full RepE pipeline:
  1. Activation extraction via PyTorch forward hooks
  2. Contrastive difference computation
  3. PCA-based vector isolation (the "Reading Vector")
  4. Real-time steering via activation injection

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
    The main Representation Engineering engine.

    Provides methods for:
      - Loading the model
      - Extracting activations from contrastive prompts
      - Computing the reading vector via PCA
      - Steering generation with the reading vector
    """

    def __init__(self, model_name: str = MODEL_NAME, target_layer: int = TARGET_LAYER):
        self.model_name = model_name
        self.target_layer = target_layer
        self.model = None
        self.tokenizer = None
        self.reading_vector: Optional[torch.Tensor] = None
        self.pca_model: Optional[PCA] = None
        self.pca_explained_variance: Optional[np.ndarray] = None
        self.all_difference_vectors: Optional[np.ndarray] = None
        self._steering_hook: Optional[SteeringHook] = None
        self._model_loaded = False

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

    def _get_target_layer_module(self):
        """Get the specific layer module to hook into."""
        layers = self._get_layers()
        if self.target_layer >= len(layers):
            raise ValueError(
                f"Target layer {self.target_layer} exceeds model depth ({len(layers)} layers). "
                f"Try a value between 0 and {len(layers) - 1}."
            )
        return layers[self.target_layer]

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
    ) -> np.ndarray:
        """
        Phase 1, Step 1: Extract difference vectors from contrastive prompt pairs.

        For each pair:
          difference = activation(positive) - activation(negative)

        Args:
            prompt_pairs: List of dicts with 'positive' and 'negative' keys
            token_position: Token position to extract (-1 = last)
            show_progress: Whether to show a progress bar

        Returns:
            numpy array of shape (n_pairs, hidden_dim)
        """
        self.load_model()

        difference_vectors = []
        iterator = tqdm(prompt_pairs, desc="🧠 Extracting activations") if show_progress else prompt_pairs

        for pair in iterator:
            pos_activation = self.extract_activation(pair["positive"], token_position)
            neg_activation = self.extract_activation(pair["negative"], token_position)

            # The magic subtraction: isolate the concept direction
            diff = pos_activation - neg_activation
            difference_vectors.append(diff.numpy())

        self.all_difference_vectors = np.array(difference_vectors, dtype=np.float32)
        print(f"✅ Extracted {len(difference_vectors)} difference vectors")
        print(f"   Shape: {self.all_difference_vectors.shape}")
        return self.all_difference_vectors

    def compute_reading_vector(
        self,
        difference_vectors: Optional[np.ndarray] = None,
        n_components: int = N_PCA_COMPONENTS,
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

        Returns:
            The reading vector as a torch.Tensor of shape (hidden_dim,)
        """
        if difference_vectors is None:
            difference_vectors = self.all_difference_vectors
        if difference_vectors is None:
            raise ValueError("No difference vectors available. Run extract_contrastive_vectors first.")

        print(f"📊 Running PCA with {n_components} components on {len(difference_vectors)} vectors...")

        # Fit PCA
        self.pca_model = PCA(n_components=n_components)
        self.pca_model.fit(difference_vectors)

        # The first component is our reading vector
        reading_vector_np = self.pca_model.components_[0]
        self.reading_vector = torch.tensor(reading_vector_np, dtype=torch.float32)
        self.pca_explained_variance = self.pca_model.explained_variance_ratio_

        print(f"✅ Reading vector computed!")
        print(f"   Explained variance ratios: {[f'{v:.4f}' for v in self.pca_explained_variance]}")
        print(f"   PC1 captures {self.pca_explained_variance[0]*100:.1f}% of variance")
        print(f"   Vector norm: {torch.norm(self.reading_vector).item():.4f}")

        return self.reading_vector

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
        if self.reading_vector is None:
            raise ValueError("No reading vector to save. Run compute_reading_vector first.")

        vector_path = os.path.join(VECTORS_DIR, f"{name}_vector.pt")
        meta_path = os.path.join(VECTORS_DIR, f"{name}_metadata.json")

        torch.save(self.reading_vector, vector_path)

        meta = {
            "name": name,
            "model": self.model_name,
            "target_layer": self.target_layer,
            "hidden_dim": int(self.reading_vector.shape[0]),
            "vector_norm": float(torch.norm(self.reading_vector).item()),
            "pca_explained_variance": self.pca_explained_variance.tolist() if self.pca_explained_variance is not None else None,
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

        self.reading_vector = torch.load(vector_path, map_location="cpu", weights_only=True)
        print(f"📂 Loaded vector from: {vector_path}")
        print(f"   Shape: {self.reading_vector.shape}")
        print(f"   Norm: {torch.norm(self.reading_vector).item():.4f}")
        return self.reading_vector

    def steer(self, coefficient: float = DEFAULT_STEERING_COEFF):
        """
        Phase 2: Attach the steering hook to the model.

        This modifies the model's forward pass so that the reading vector
        is added to the hidden states at the target layer.

        Args:
            coefficient: Scaling factor.
                         Positive → amplify concept (e.g., more honest)
                         Negative → suppress/invert concept (e.g., more deceptive)
                         Zero → no steering (baseline)
        """
        self.load_model()
        self.unsteer()  # Remove any existing steering

        if self.reading_vector is None:
            raise ValueError("No reading vector available. Extract or load one first.")

        self._steering_hook = SteeringHook(self.reading_vector, coefficient)
        self._steering_hook.register(self._get_target_layer_module())
        print(f"🎯 Steering active: coefficient = {coefficient:+.2f}")

    def unsteer(self):
        """Remove any active steering hook."""
        if self._steering_hook is not None:
            self._steering_hook.remove()
            self._steering_hook = None

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
            The generated text (prompt + completion)
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

    def compare_steered_outputs(
        self,
        prompt: str,
        coefficients: List[float] = [-5.0, -3.0, 0.0, 3.0, 5.0],
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> List[Dict]:
        """
        Generate outputs at various steering coefficients for comparison.

        Returns a list of dicts with 'coefficient' and 'output' keys.
        """
        results = []

        for coeff in coefficients:
            if coeff == 0.0:
                self.unsteer()
                label = "baseline (no steering)"
            else:
                self.steer(coeff)
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

    def project_onto_vector(self, text: str) -> float:
        """
        'Read the model's mind' — project a text's activation onto the
        reading vector to measure how much the model's internal state
        aligns with the concept.

        Returns a scalar: positive = aligned with concept, negative = opposed.
        """
        if self.reading_vector is None:
            raise ValueError("No reading vector available.")

        activation = self.extract_activation(text)
        # Normalize the reading vector for a clean projection
        normalized_vec = self.reading_vector / torch.norm(self.reading_vector)
        projection = torch.dot(activation, normalized_vec).item()
        return projection

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

    def get_pca_projection_data(self) -> Optional[Dict]:
        """
        Get 2D PCA projection data for visualization.
        Projects all difference vectors onto the first two principal components.
        """
        if self.pca_model is None or self.all_difference_vectors is None:
            return None

        projections = self.pca_model.transform(self.all_difference_vectors)
        return {
            "pc1": projections[:, 0].tolist(),
            "pc2": projections[:, 1].tolist(),
            "explained_variance": self.pca_explained_variance.tolist(),
        }

    def get_status(self) -> Dict:
        """Get the current status of the engine."""
        return {
            "model_loaded": self._model_loaded,
            "model_name": self.model_name,
            "target_layer": self.target_layer,
            "device": DEVICE,
            "has_reading_vector": self.reading_vector is not None,
            "vector_norm": float(torch.norm(self.reading_vector).item()) if self.reading_vector is not None else None,
            "is_steering": self._steering_hook is not None,
            "steering_coefficient": self._steering_hook.coefficient if self._steering_hook is not None else None,
            "pca_variance_explained": self.pca_explained_variance.tolist() if self.pca_explained_variance is not None else None,
        }
