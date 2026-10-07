"""
app.py — Interactive Web Dashboard for LLM Mind Reader (v2.0)

A Flask + SocketIO server that provides a real-time interface for:
  - Multi-concept vector extraction (honesty, sycophancy, power-seeking, risk-aversion)
  - Neural activation heatmaps
  - Concept probe classifier training & evaluation
  - Steered generation with streaming token output
  - Multi-layer simultaneous steering
  - 3D PCA visualization
  - Side-by-side comparison at multiple coefficients

Run:
    python app.py
    # Open http://localhost:5000
"""

import os
import json
import threading
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from repe_engine import RepEEngine
from contrastive_prompts import ALL_CONCEPTS, HONESTY_PROMPTS
from config import FLASK_HOST, FLASK_PORT, TARGET_LAYER, VECTORS_DIR

# ── Flask App Setup ───────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "repe-mind-reader-2024"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Global Engine Instance ────────────────────────────────────────
engine = RepEEngine(target_layer=TARGET_LAYER)

# ── Routes ────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    """Get current engine status."""
    status_data = engine.get_status()
    # Check if any saved vectors exist
    status_data["saved_vectors"] = engine.get_available_saved_vectors()
    return jsonify(status_data)


@app.route("/api/concepts")
def concepts():
    """List available concepts and their metadata."""
    result = {}
    for name, info in ALL_CONCEPTS.items():
        result[name] = {
            "description": info["description"],
            "positive_label": info["positive_label"],
            "negative_label": info["negative_label"],
            "color": info["color"],
            "n_prompts": len(info["prompts"]),
        }
    return jsonify(result)


@app.route("/api/saved-vectors")
def saved_vectors():
    """List all saved vectors."""
    return jsonify(engine.get_available_saved_vectors())


# ── SocketIO Events ──────────────────────────────────────────────


@socketio.on("connect")
def handle_connect():
    print("🌐 Client connected")
    emit("status_update", engine.get_status())


@socketio.on("load_model")
def handle_load_model():
    """Load the model (first time takes a while)."""
    try:
        emit("log", {"message": "🔄 Loading model... This may take 1-2 minutes on first run.", "type": "info"})
        engine.load_model()
        emit("log", {"message": "✅ Model loaded successfully!", "type": "success"})
        emit("status_update", engine.get_status())
    except Exception as e:
        emit("log", {"message": f"❌ Error loading model: {str(e)}", "type": "error"})


@socketio.on("extract_vector")
def handle_extract_vector(data):
    """Extract a reading vector for a given concept."""
    try:
        concept_name = data.get("concept", "honesty")
        n_prompts = data.get("n_prompts", None)

        if concept_name not in ALL_CONCEPTS:
            emit("log", {"message": f"❌ Unknown concept: {concept_name}", "type": "error"})
            return

        concept_info = ALL_CONCEPTS[concept_name]
        prompts = concept_info["prompts"]
        if n_prompts is not None:
            prompts = prompts[:n_prompts]

        emit("log", {
            "message": f"🧠 Starting {concept_name} extraction with {len(prompts)} prompt pairs...",
            "type": "info",
        })
        emit("extraction_progress", {"current": 0, "total": len(prompts)})

        # Extract activations with progress updates
        engine.load_model()
        engine.active_concept = concept_name

        import numpy as np
        import torch
        difference_vectors = []

        for i, pair in enumerate(prompts):
            pos_act = engine.extract_activation(pair["positive"])
            neg_act = engine.extract_activation(pair["negative"])
            diff = pos_act - neg_act
            difference_vectors.append(diff.numpy())

            if (i + 1) % 5 == 0 or i == len(prompts) - 1:
                emit("extraction_progress", {
                    "current": i + 1,
                    "total": len(prompts),
                })

        engine.all_difference_vectors[concept_name] = np.array(difference_vectors, dtype=np.float32)

        # Compute PCA
        emit("log", {"message": "📊 Running PCA...", "type": "info"})
        reading_vector = engine.compute_reading_vector(concept_name=concept_name)
        reading_vector = engine._ensure_correct_sign(reading_vector, prompts)
        engine.reading_vectors[concept_name] = reading_vector

        # Save
        engine.save_vector(concept_name, metadata={"n_prompts": len(prompts)})

        # Get PCA projection data for visualization
        pca_data = engine.get_pca_projection_data(concept_name)

        emit("log", {
            "message": f"✅ {concept_name} vector extracted! PC1 explains {engine.pca_explained_variances[concept_name][0]*100:.1f}% of variance.",
            "type": "success",
        })
        emit("extraction_complete", {
            "concept": concept_name,
            "pca_data": pca_data,
            "variance_explained": engine.pca_explained_variances[concept_name].tolist(),
            "vector_norm": float(torch.norm(engine.reading_vectors[concept_name]).item()),
        })
        emit("status_update", engine.get_status())

    except Exception as e:
        emit("log", {"message": f"❌ Extraction error: {str(e)}", "type": "error"})
        import traceback
        traceback.print_exc()


@socketio.on("load_vector")
def handle_load_vector(data):
    """Load a previously saved vector."""
    try:
        name = data.get("name", "honesty")
        engine.load_model()
        engine.load_vector(name)
        emit("log", {"message": f"📂 Loaded vector: {name}", "type": "success"})
        emit("status_update", engine.get_status())
    except Exception as e:
        emit("log", {"message": f"❌ Error loading vector: {str(e)}", "type": "error"})


@socketio.on("generate")
def handle_generate(data):
    """Generate text with optional steering (streaming)."""
    try:
        prompt = data.get("prompt", "")
        coefficient = data.get("coefficient", 0.0)
        max_tokens = data.get("max_tokens", 200)
        concept = data.get("concept", engine.active_concept)
        streaming = data.get("streaming", True)

        if not prompt.strip():
            emit("log", {"message": "⚠️ Empty prompt", "type": "warning"})
            return

        engine.load_model()
        engine.active_concept = concept

        # Apply steering
        if coefficient != 0.0 and concept in engine.reading_vectors:
            engine.steer(coefficient, concept_name=concept)
            emit("log", {
                "message": f"🎯 Generating with {concept} steering c={coefficient:+.1f}",
                "type": "info",
            })
        else:
            engine.unsteer()
            emit("log", {"message": "💬 Generating baseline (no steering)", "type": "info"})

        if streaming:
            # Stream tokens one at a time
            full_output = ""
            for token in engine.generate_streaming(prompt, max_new_tokens=max_tokens):
                full_output += token
                emit("generation_token", {"token": token})
            output = full_output
        else:
            output = engine.generate(prompt, max_new_tokens=max_tokens)

        # Compute mind-reading projection if vector available
        projection = None
        if concept in engine.reading_vectors:
            full_text = prompt + " " + output
            projection = engine.project_onto_vector(full_text, concept_name=concept)

        engine.unsteer()

        emit("generation_complete", {
            "prompt": prompt,
            "output": output,
            "coefficient": coefficient,
            "concept": concept,
            "projection": projection,
        })

    except Exception as e:
        emit("log", {"message": f"❌ Generation error: {str(e)}", "type": "error"})
        import traceback
        traceback.print_exc()


@socketio.on("generate_multi_layer")
def handle_generate_multi_layer(data):
    """Generate with multi-layer steering."""
    try:
        prompt = data.get("prompt", "")
        layer_coefficients = data.get("layer_coefficients", {})
        max_tokens = data.get("max_tokens", 200)
        concept = data.get("concept", engine.active_concept)

        if not prompt.strip():
            emit("log", {"message": "⚠️ Empty prompt", "type": "warning"})
            return

        engine.load_model()

        # Convert string keys to int
        layer_coeffs = {int(k): float(v) for k, v in layer_coefficients.items()}

        if any(v != 0 for v in layer_coeffs.values()) and concept in engine.reading_vectors:
            engine.steer_multi_layer(layer_coeffs, concept_name=concept)
            active = {k: v for k, v in layer_coeffs.items() if v != 0}
            emit("log", {"message": f"🎯 Multi-layer steering: {active}", "type": "info"})
        else:
            engine.unsteer()

        output = engine.generate(prompt, max_new_tokens=max_tokens)
        engine.unsteer()

        emit("generation_complete", {
            "prompt": prompt,
            "output": output,
            "coefficient": 0.0,
            "concept": concept,
            "multi_layer": True,
        })

    except Exception as e:
        emit("log", {"message": f"❌ Multi-layer generation error: {str(e)}", "type": "error"})
        import traceback
        traceback.print_exc()


@socketio.on("mind_read")
def handle_mind_read(data):
    """Project text onto the reading vector to 'read the model's mind'."""
    try:
        text = data.get("text", "")
        concept = data.get("concept", engine.active_concept)
        if not text.strip():
            return

        engine.load_model()
        if concept not in engine.reading_vectors:
            engine.load_vector(concept)

        projection = engine.project_onto_vector(text, concept_name=concept)

        emit("mind_read_result", {
            "text": text,
            "projection": projection,
            "concept": concept,
        })

    except Exception as e:
        emit("log", {"message": f"❌ Mind reading error: {str(e)}", "type": "error"})


@socketio.on("activation_heatmap")
def handle_activation_heatmap(data):
    """Extract activations from all layers for heatmap visualization."""
    try:
        text = data.get("text", "")
        if not text.strip():
            emit("log", {"message": "⚠️ Empty text", "type": "warning"})
            return

        emit("log", {"message": "🔥 Extracting activations across all layers...", "type": "info"})
        engine.load_model()
        heatmap_data = engine.extract_all_layer_activations(text)

        emit("heatmap_complete", heatmap_data)
        emit("log", {
            "message": f"✅ Heatmap ready: {heatmap_data['n_layers']} layers × {heatmap_data['n_tokens']} tokens",
            "type": "success",
        })

    except Exception as e:
        emit("log", {"message": f"❌ Heatmap error: {str(e)}", "type": "error"})
        import traceback
        traceback.print_exc()


@socketio.on("train_probe")
def handle_train_probe(data):
    """Train a linear probe classifier for a concept."""
    try:
        concept = data.get("concept", "honesty")
        n_prompts = data.get("n_prompts", None)

        if concept not in ALL_CONCEPTS:
            emit("log", {"message": f"❌ Unknown concept: {concept}", "type": "error"})
            return

        prompts = ALL_CONCEPTS[concept]["prompts"]
        if n_prompts is not None:
            prompts = prompts[:n_prompts]

        emit("log", {"message": f"🔬 Training probe for '{concept}'...", "type": "info"})
        engine.load_model()

        metrics = engine.train_probe(prompts, concept_name=concept)

        emit("probe_complete", {
            "concept": concept,
            "metrics": metrics,
        })
        emit("log", {
            "message": f"✅ Probe accuracy: {metrics['accuracy']:.1%} | F1: {metrics['f1']:.3f}",
            "type": "success",
        })
        emit("status_update", engine.get_status())

    except Exception as e:
        emit("log", {"message": f"❌ Probe training error: {str(e)}", "type": "error"})
        import traceback
        traceback.print_exc()


@socketio.on("probe_text")
def handle_probe_text(data):
    """Classify text using the trained probe."""
    try:
        text = data.get("text", "")
        concept = data.get("concept", "honesty")

        if not text.strip():
            return

        engine.load_model()
        result = engine.probe_text(text, concept_name=concept)

        emit("probe_result", {
            "text": text,
            "concept": concept,
            **result,
        })

    except Exception as e:
        emit("log", {"message": f"❌ Probe classification error: {str(e)}", "type": "error"})


@socketio.on("scan_layers")
def handle_scan_layers():
    """Scan all layers to find the best one for extraction."""
    try:
        emit("log", {"message": "🔍 Scanning layers... This may take a few minutes.", "type": "info"})
        engine.load_model()
        scan_data = engine.get_layer_scan_data(HONESTY_PROMPTS, n_pairs=5)

        emit("layer_scan_complete", scan_data)
        emit("log", {
            "message": f"✅ Layer scan complete. Best layer: {scan_data['best_layer']}",
            "type": "success",
        })
    except Exception as e:
        emit("log", {"message": f"❌ Layer scan error: {str(e)}", "type": "error"})


@socketio.on("compare_outputs")
def handle_compare_outputs(data):
    """Generate outputs at multiple steering coefficients."""
    try:
        prompt = data.get("prompt", "")
        coefficients = data.get("coefficients", [-5.0, -3.0, 0.0, 3.0, 5.0])
        max_tokens = data.get("max_tokens", 150)
        concept = data.get("concept", engine.active_concept)

        if not prompt.strip():
            emit("log", {"message": "⚠️ Empty prompt", "type": "warning"})
            return

        engine.load_model()
        if concept not in engine.reading_vectors:
            engine.load_vector(concept)

        results = []
        for i, coeff in enumerate(coefficients):
            emit("log", {
                "message": f"🔄 Generating {i+1}/{len(coefficients)}: c={coeff:+.1f}",
                "type": "info",
            })

            if coeff == 0.0:
                engine.unsteer()
            else:
                engine.steer(coeff, concept_name=concept)

            output = engine.generate(prompt, max_new_tokens=max_tokens)

            results.append({
                "coefficient": coeff,
                "output": output,
            })

            emit("compare_progress", {
                "current": i + 1,
                "total": len(coefficients),
                "result": results[-1],
            })

        engine.unsteer()

        emit("compare_complete", {
            "prompt": prompt,
            "concept": concept,
            "results": results,
        })

    except Exception as e:
        emit("log", {"message": f"❌ Comparison error: {str(e)}", "type": "error"})
        import traceback
        traceback.print_exc()


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🧠 LLM Mind Reader — Interactive Dashboard v2.0")
    print("=" * 60)
    print(f"\n  Open: http://localhost:{FLASK_PORT}")
    print(f"  Press Ctrl+C to stop\n")

    socketio.run(app, host=FLASK_HOST, port=FLASK_PORT, debug=False)
