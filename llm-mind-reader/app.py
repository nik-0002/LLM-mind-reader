"""
app.py — Interactive Web Dashboard for LLM Mind Reader (v3.0)

Flask + SocketIO server:
  /      the original dashboard (templates/index.html, if you have it)
  /lab   redirects to the Analyze tab

Run:
    python app.py        # http://localhost:5000
"""

import os
import threading
import traceback
from functools import wraps

from flask import Flask, render_template, jsonify, redirect
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from repe_engine import RepEEngine
from contrastive_prompts import ALL_CONCEPTS
from config import FLASK_HOST, FLASK_PORT, TARGET_LAYER, DEFAULT_STEERING_COEFF, COMPARE_COEFFICIENTS

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("REPE_SECRET", os.urandom(16).hex())
CORS(app, origins=[r"http://localhost:\d+", r"http://127\.0\.0\.1:\d+"])
socketio = SocketIO(app, cors_allowed_origins=[f"http://localhost:{FLASK_PORT}", f"http://127.0.0.1:{FLASK_PORT}"],
                    async_mode="threading")

engine = RepEEngine(target_layer=TARGET_LAYER)

# One model, one set of hooks: serialise anything that touches the engine so two browser
# tabs can't steer each other's generations.
_engine_lock = threading.RLock()


def guarded(label):
    """Serialise access to the engine and report errors to the client log."""
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            with _engine_lock:
                try:
                    return fn(*a, **kw)
                except Exception as e:                       # noqa: BLE001
                    traceback.print_exc()
                    engine.unsteer()
                    emit("log", {"message": f"❌ {label}: {e}", "type": "error"})
        return wrapper
    return deco


def log(msg, kind="info"):
    emit("log", {"message": msg, "type": kind})


def ensure_vector(concept):
    if concept not in engine.reading_vectors:
        prev = engine.active_concept
        engine.load_vector(concept)
        engine.active_concept = prev


# ── HTTP routes ───────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/lab")
def lab():
    return redirect("/#analyze")        # the old lab page was merged into the main dashboard


@app.route("/api/status")
def status():
    data = engine.get_status()
    data["saved_vectors"] = engine.get_available_saved_vectors()
    return jsonify(data)


@app.route("/api/concepts")
def concepts():
    return jsonify({n: {"description": i["description"], "positive_label": i["positive_label"],
                        "negative_label": i["negative_label"], "color": i["color"],
                        "n_prompts": len(i["prompts"])} for n, i in ALL_CONCEPTS.items()})


@app.route("/api/saved-vectors")
def saved_vectors():
    return jsonify(engine.get_available_saved_vectors())


@app.route("/api/defaults")
def defaults():
    return jsonify({"default_coeff": DEFAULT_STEERING_COEFF, "compare_coeffs": COMPARE_COEFFICIENTS})


# ── Socket events ─────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    emit("status_update", engine.get_status())


@socketio.on("load_model")
@guarded("Model load")
def on_load_model(*_):
    log("🔄 Loading model... may take 1-2 minutes on first run.")
    engine.load_model()
    log("✅ Model loaded!", "success")
    emit("status_update", engine.get_status())


@socketio.on("extract_vector")
@guarded("Extraction")
def on_extract_vector(data):
    concept = data.get("concept", "honesty")
    n_prompts = data.get("n_prompts")
    method = data.get("method")
    if concept not in ALL_CONCEPTS:
        return log(f"❌ Unknown concept: {concept}", "error")

    prompts = ALL_CONCEPTS[concept]["prompts"]
    if n_prompts:
        prompts = prompts[:n_prompts]
    log(f"🧠 Extracting '{concept}' from {len(prompts)} pairs (all layers in one pass)...")
    emit("extraction_progress", {"current": 0, "total": len(prompts)})

    engine.load_model()
    engine.active_concept = concept

    def progress(i, n):
        if i % 5 == 0 or i == n:
            emit("extraction_progress", {"current": i, "total": n})

    engine.extract_pair_activations(prompts, concept, show_progress=False, progress_cb=progress)
    log("📊 Computing direction, scale and per-layer vectors...")
    engine.compute_reading_vector(concept_name=concept, method=method)
    engine.save_vector(concept, metadata={"n_prompts": len(prompts), "concept": concept})

    var = engine.pca_explained_variances[concept].tolist()
    log(f"✅ '{concept}' ready — PC1/direction energy {var[0] * 100:.1f}%, "
        f"steer scale {engine.steer_scales[concept]:.2f}", "success")
    emit("extraction_complete", {
        "concept": concept,
        "pca_data": engine.get_pca_projection_data(concept),
        "scatter": engine.get_activation_scatter(concept),
        "variance_explained": var,
        "vector_norm": float(engine.reading_vectors[concept].norm().item()),
        "steer_scale": engine.steer_scales[concept],
        "calibration": engine.calibrations.get(concept),
    })
    emit("status_update", engine.get_status())


@socketio.on("load_vector")
@guarded("Load vector")
def on_load_vector(data):
    name = data.get("name", "honesty")
    engine.load_model()
    engine.load_vector(name)
    log(f"📂 Loaded vector: {name}", "success")
    emit("status_update", engine.get_status())


@socketio.on("generate")
@guarded("Generation")
def on_generate(data):
    prompt = data.get("prompt", "")
    coeff = float(data.get("coefficient", 0.0))
    max_tokens = int(data.get("max_tokens", 200))
    concept = data.get("concept", engine.active_concept)
    streaming = data.get("streaming", True)
    seed = data.get("seed")
    if not prompt.strip():
        return log("⚠️ Empty prompt", "warning")

    engine.load_model()
    if coeff != 0.0:
        ensure_vector(concept)
        engine.active_concept = concept
        engine.steer(coeff, concept_name=concept)
        log(f"🎯 {concept} steering c={coeff:+.2f}")
    else:
        engine.unsteer()
        log("💬 Baseline (no steering)")

    try:
        if streaming:
            parts = []
            for tok in engine.generate_streaming(prompt, max_new_tokens=max_tokens, seed=seed):
                parts.append(tok)
                emit("generation_token", {"token": tok})
            output = "".join(parts)
        else:
            output = engine.generate(prompt, max_new_tokens=max_tokens, seed=seed)
    finally:
        engine.unsteer()

    reading = engine.read_concept(prompt + " " + output, concept) if concept in engine.reading_vectors else None
    emit("generation_complete", {
        "prompt": prompt, "output": output, "coefficient": coeff, "concept": concept,
        "projection": reading["projection"] if reading else None,
        "normalized": reading["normalized"] if reading else None,
    })


@socketio.on("generate_multi_layer")
@guarded("Multi-layer generation")
def on_generate_multi_layer(data):
    prompt = data.get("prompt", "")
    concept = data.get("concept", engine.active_concept)
    max_tokens = int(data.get("max_tokens", 200))
    coeffs = {int(k): float(v) for k, v in data.get("layer_coefficients", {}).items()}
    if not prompt.strip():
        return log("⚠️ Empty prompt", "warning")

    engine.load_model()
    ensure_vector(concept)
    try:
        if any(v != 0 for v in coeffs.values()):
            engine.steer_multi_layer(coeffs, concept_name=concept)
        else:
            engine.unsteer()
        output = engine.generate(prompt, max_new_tokens=max_tokens, seed=data.get("seed"))
    finally:
        engine.unsteer()
    emit("generation_complete", {"prompt": prompt, "output": output, "coefficient": 0.0,
                                 "concept": concept, "multi_layer": True})


@socketio.on("mind_read")
@guarded("Mind reading")
def on_mind_read(data):
    text = data.get("text", "")
    concept = data.get("concept", engine.active_concept)
    if not text.strip():
        return
    engine.load_model()
    ensure_vector(concept)
    r = engine.read_concept(text, concept)
    emit("mind_read_result", {"text": text, **r})


@socketio.on("token_scan")
@guarded("Token scan")
def on_token_scan(data):
    text = data.get("text", "")
    concept = data.get("concept", engine.active_concept)
    layer = data.get("layer")
    if not text.strip():
        return log("⚠️ Empty text", "warning")
    engine.load_model()
    ensure_vector(concept)
    emit("token_scan_result", engine.scan_tokens(text, concept, layer=int(layer) if layer not in (None, "") else None))


@socketio.on("activation_heatmap")
@guarded("Heatmap")
def on_activation_heatmap(data):
    text = data.get("text", "")
    if not text.strip():
        return log("⚠️ Empty text", "warning")
    engine.load_model()
    hm = engine.extract_all_layer_activations(text)
    emit("heatmap_complete", hm)
    log(f"✅ Heatmap: {hm['n_layers']} layers × {hm['n_tokens']} tokens", "success")


@socketio.on("train_probe")
@guarded("Probe training")
def on_train_probe(data):
    concept = data.get("concept", "honesty")
    if concept not in ALL_CONCEPTS:
        return log(f"❌ Unknown concept: {concept}", "error")
    prompts = ALL_CONCEPTS[concept]["prompts"]
    if data.get("n_prompts"):
        prompts = prompts[: data["n_prompts"]]
    engine.load_model()
    metrics = engine.train_probe(prompts, concept_name=concept)
    emit("probe_complete", {"concept": concept, "metrics": metrics})
    log(f"✅ Probe acc {metrics['accuracy']:.1%} | AUROC {metrics['auroc']:.3f} (pair-grouped split)", "success")
    emit("status_update", engine.get_status())


@socketio.on("probe_text")
@guarded("Probe")
def on_probe_text(data):
    text, concept = data.get("text", ""), data.get("concept", "honesty")
    if text.strip():
        engine.load_model()
        emit("probe_result", {"text": text, "concept": concept, **engine.probe_text(text, concept)})


@socketio.on("scan_layers")
@guarded("Layer sweep")
def on_scan_layers(data=None):
    data = data or {}
    concept = data.get("concept", "honesty")
    n_pairs = int(data.get("n_pairs", 40))
    log(f"🔍 Held-out layer sweep for '{concept}' on {n_pairs} pairs...")
    engine.load_model()
    res = engine.layer_sweep(ALL_CONCEPTS[concept]["prompts"][:n_pairs], concept)
    emit("layer_scan_complete", res)
    log(f"✅ Best layer {res['best_layer']} (held-out AUROC {res['auroc'][res['best_layer']]:.3f})", "success")


@socketio.on("concept_similarity")
@guarded("Similarity")
def on_concept_similarity(*_):
    engine.load_model()
    for name in ALL_CONCEPTS:
        try:
            ensure_vector(name)
        except FileNotFoundError:
            pass
    emit("concept_similarity_result", engine.concept_similarity())


@socketio.on("activation_scatter")
@guarded("Scatter")
def on_activation_scatter(data):
    concept = data.get("concept", engine.active_concept)
    sc = engine.get_activation_scatter(concept)
    if sc is None:
        return log("ℹ️ Extract the vector in this session first (scatter needs the raw activations).", "warning")
    emit("activation_scatter_result", {"concept": concept, **sc})


@socketio.on("compare_outputs")
@guarded("Comparison")
def on_compare_outputs(data):
    prompt = data.get("prompt", "")
    coeffs = data.get("coefficients", COMPARE_COEFFICIENTS)
    max_tokens = int(data.get("max_tokens", 150))
    concept = data.get("concept", engine.active_concept)
    seed = data.get("seed", 42)
    if not prompt.strip():
        return log("⚠️ Empty prompt", "warning")

    engine.load_model()
    ensure_vector(concept)
    results = []
    try:
        for i, c in enumerate(coeffs):
            log(f"🔄 {i + 1}/{len(coeffs)}: c={c:+.2f}")
            engine.steer(c, concept) if c != 0 else engine.unsteer()
            out = engine.generate(prompt, max_new_tokens=max_tokens, seed=seed)   # same seed for every c
            engine.unsteer()
            reading = engine.read_concept(prompt + out, concept)
            results.append({"coefficient": c, "output": out, "normalized": reading["normalized"]})
            emit("compare_progress", {"current": i + 1, "total": len(coeffs), "result": results[-1]})
    finally:
        engine.unsteer()
    emit("compare_complete", {"prompt": prompt, "concept": concept, "results": results})


if __name__ == "__main__":
    print("\n" + "=" * 60 + "\n  🧠 LLM Mind Reader — Dashboard v3.0\n" + "=" * 60)
    print(f"\n  Open: http://localhost:{FLASK_PORT}   \n  Ctrl+C to stop\n")
    socketio.run(app, host=FLASK_HOST, port=FLASK_PORT, debug=False, allow_unsafe_werkzeug=True)
