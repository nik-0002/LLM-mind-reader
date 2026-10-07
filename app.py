"""
app.py — Interactive Web Dashboard for LLM Mind Reader

A Flask + SocketIO server that provides a real-time interface for:
  - Extracting reading vectors
  - Visualizing PCA projections
  - Steering generation with a live coefficient slider
  - Projecting arbitrary text onto the reading vector ("mind reading")

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
from contrastive_prompts import HONESTY_PROMPTS
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
    # Check if a saved vector exists
    vector_path = os.path.join(VECTORS_DIR, "honesty_vector.pt")
    status_data["saved_vector_exists"] = os.path.exists(vector_path)
    return jsonify(status_data)


@app.route("/api/saved-vectors")
def saved_vectors():
    """List all saved vectors."""
    vectors = []
    if os.path.exists(VECTORS_DIR):
        for f in os.listdir(VECTORS_DIR):
            if f.endswith("_metadata.json"):
                with open(os.path.join(VECTORS_DIR, f)) as fh:
                    meta = json.load(fh)
                    vectors.append(meta)
    return jsonify(vectors)


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
    """Extract the honesty reading vector."""
    try:
        n_prompts = data.get("n_prompts", 100)
        prompts = HONESTY_PROMPTS[:n_prompts]

        emit("log", {"message": f"🧠 Starting extraction with {len(prompts)} prompt pairs...", "type": "info"})
        emit("extraction_progress", {"current": 0, "total": len(prompts)})

        # Extract activations with progress updates
        engine.load_model()

        # Manual extraction loop for progress reporting
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

        engine.all_difference_vectors = np.array(difference_vectors, dtype=np.float32)

        # Compute PCA
        emit("log", {"message": "📊 Running PCA...", "type": "info"})
        reading_vector = engine.compute_reading_vector()
        reading_vector = engine._ensure_correct_sign(reading_vector, prompts)
        engine.reading_vector = reading_vector

        # Save
        engine.save_vector("honesty", metadata={"n_prompts": len(prompts)})

        # Get PCA projection data for visualization
        pca_data = engine.get_pca_projection_data()

        emit("log", {
            "message": f"✅ Reading vector extracted! PC1 explains {engine.pca_explained_variance[0]*100:.1f}% of variance.",
            "type": "success"
        })
        emit("extraction_complete", {
            "pca_data": pca_data,
            "variance_explained": engine.pca_explained_variance.tolist(),
            "vector_norm": float(torch.norm(engine.reading_vector).item()),
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
    """Generate text with optional steering."""
    try:
        prompt = data.get("prompt", "")
        coefficient = data.get("coefficient", 0.0)
        max_tokens = data.get("max_tokens", 200)

        if not prompt.strip():
            emit("log", {"message": "⚠️ Empty prompt", "type": "warning"})
            return

        engine.load_model()

        # Apply steering
        if coefficient != 0.0 and engine.reading_vector is not None:
            engine.steer(coefficient)
            emit("log", {
                "message": f"🎯 Generating with steering coefficient = {coefficient:+.1f}",
                "type": "info"
            })
        else:
            engine.unsteer()
            emit("log", {"message": "💬 Generating baseline (no steering)", "type": "info"})

        output = engine.generate(prompt, max_new_tokens=max_tokens)

        # Also compute mind-reading projection if vector available
        projection = None
        if engine.reading_vector is not None:
            full_text = prompt + " " + output
            projection = engine.project_onto_vector(full_text)

        engine.unsteer()

        emit("generation_complete", {
            "prompt": prompt,
            "output": output,
            "coefficient": coefficient,
            "projection": projection,
        })

    except Exception as e:
        emit("log", {"message": f"❌ Generation error: {str(e)}", "type": "error"})
        import traceback
        traceback.print_exc()


@socketio.on("mind_read")
def handle_mind_read(data):
    """Project text onto the reading vector to 'read the model's mind'."""
    try:
        text = data.get("text", "")
        if not text.strip():
            return

        engine.load_model()
        if engine.reading_vector is None:
            engine.load_vector("honesty")

        projection = engine.project_onto_vector(text)

        emit("mind_read_result", {
            "text": text,
            "projection": projection,
        })

    except Exception as e:
        emit("log", {"message": f"❌ Mind reading error: {str(e)}", "type": "error"})


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
            "type": "success"
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

        if not prompt.strip():
            emit("log", {"message": "⚠️ Empty prompt", "type": "warning"})
            return

        engine.load_model()
        if engine.reading_vector is None:
            engine.load_vector("honesty")

        results = []
        for i, coeff in enumerate(coefficients):
            emit("log", {
                "message": f"🔄 Generating {i+1}/{len(coefficients)}: coefficient = {coeff:+.1f}",
                "type": "info"
            })

            if coeff == 0.0:
                engine.unsteer()
            else:
                engine.steer(coeff)

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
            "results": results,
        })

    except Exception as e:
        emit("log", {"message": f"❌ Comparison error: {str(e)}", "type": "error"})
        import traceback
        traceback.print_exc()


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🧠 LLM Mind Reader — Interactive Dashboard")
    print("=" * 60)
    print(f"\n  Open: http://localhost:{FLASK_PORT}")
    print(f"  Press Ctrl+C to stop\n")

    socketio.run(app, host=FLASK_HOST, port=FLASK_PORT, debug=False)
