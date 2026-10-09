# LLM Mind Reader & Controller
### Representation Engineering (RepE) — after Zou et al., 2023

**Read** and **control** an LLM's internal representations without retraining.
Extract the direction for a concept (honesty, sycophancy, power-seeking, risk-aversion) from a
model's hidden states, use it to *read* what the model is "thinking" token by token, then add it back
into the residual stream to *steer* generation — and **measure whether any of it actually works**.

## Quick start

```bash
pip install -r requirements.txt
huggingface-cli login                       # Llama-3.2-1B is gated

# 0. sanity-check the maths (no GPU / model needed)
python tests/test_vector_math.py

# 1. find the best layer (held-out, not a guess)
python extract_vectors.py --scan-layers

# 2. extract vectors (all four concepts)
python extract_vectors.py --concept all --layer <best>

# 3. read: per-token lie detector
python lie_detector.py "I never make mistakes, and everything is perfect."

# 4. control: steer, with a random-direction control
python steer_model.py --prompt "Is homeopathy effective?" --compare --control

# 5. evaluate (writes JSON + PNG to results/)
python evaluation.py layers   --concept honesty
python evaluation.py reading  --concept honesty --all-layers
python evaluation.py steering --concept honesty
python evaluation.py similarity

# 6. dashboards
python app.py            # http://localhost:5000  (dashboard: Setup · Extract · Steer · Compare · Mind Read · Analyze)
```

## How it works

| Phase | What happens |
|---|---|
| **Read** | Run contrastive pairs (same scenario, "be honest" vs "be deceptive") through the model. Capture the residual stream at the last token — **all layers in one forward pass**. Subtract (pos − neg). |
| **Isolate** | Reduce the difference vectors to a single unit direction: RepE-style PCA (random pair-order flips, then PC1) or difference-of-means. The sign is aligned to the mean difference. |
| **Calibrate** | Record the typical positive/negative projection (→ a 0…1 *normalized score*) and the typical gap along the direction (→ the **steer scale**). |
| **Steer** | During generation add `coefficient × scale × direction` to the residual stream at the target layer. `+1` ≈ one typical negative→positive step. |
| **Evaluate** | Held-out layer sweep, true/false-statement transfer vs. a random-direction null, steering dose-response with a random-vector control and a fluency (NLL) cost curve. |

## What changed in v3 (and why)

I audited the v2 pipeline and fixed several methodological issues. These are worth understanding — they're the
kind of thing reviewers probe.

| Issue in v2 | Why it matters | v3 |
|---|---|---|
| **Steering was probably a near no-op.** PCA components have unit norm; mid-layer residual streams typically have norms in the tens–hundreds. | `coefficient=3` perturbs activations by a few percent at most. "No effect" would be mis-read as "the vector doesn't work". | Vectors carry a `steer_scale` (mean pos−neg gap along the direction). Coefficients are now in interpretable *gap units*; sweep ≈ −2…+2. |
| **Centered PCA on un-flipped differences.** sklearn's PCA subtracts the mean — and for pair differences the mean *is* the concept. | PC1 can latch onto nuisance variance instead of the concept (reproduced in `tests/test_vector_math.py`). | Random pair-order flips (as in the paper) + a difference-of-means baseline; `cos(direction, mean-diff)` is reported. |
| **Layer scan = norm of the difference.** | Residual norms grow with depth, so it favours late layers regardless of the concept. | Held-out sweep: fit on train pairs, score AUROC / pair-accuracy / d′ on unseen pairs. |
| **Probe split leaked pairs.** | A prompt's twin in the training set inflates test accuracy. | Split by *pair*; standardised features; AUROC; cosine between probe weights and the reading vector. |
| Multi-layer steering reused the layer-8 vector at other layers | A direction found at layer 8 isn't "the same" direction at layer 12. | Per-layer vectors + scales, used automatically by `steer_multi_layer`. |
| Streaming repetition penalty divided *negative* logits | Makes repeated tokens *more* likely. | HF-consistent fix. |
| Reading activations while a steering hook was attached | Silently contaminated measurements. | Extraction functions always detach steering first. |
| Server bound to `0.0.0.0`, CORS `*`, hard-coded secret, no locking | A shared GPU model with global hooks — two tabs could steer each other. | Localhost + restricted origins, random secret, a lock around all engine access. |
| Mind Read meter never updated: Steer and Mind Read tabs shared element ids | The second meter was dead UI. | Distinct ids per meter; scores are calibrated 0…1 instead of raw projection ÷ 20. |
| Different random samples per coefficient | Differences in a comparison could be sampling noise. | Same seed for every coefficient in compare/eval modes. |

### Also new
- **Redesigned dashboard** — minimal pastel-monochrome UI (single periwinkle hue, light/dark), concept selector, calibrated reader meters, token scan, layer heatmap, probe and similarity views.
- **`lie_detector.py`** — per-token concept readout (BOS excluded: attention-sink tokens have huge activations).
- **`evaluation.py`** — four experiments with JSON + PNG outputs and a *random-direction control*.
- **`eval_data.py`** — 40 held-out true/false pairs and 8 steering prompts, never used for extraction.
- **`vector_math.py`** — all the linear algebra as pure numpy, unit-tested without a GPU.
- **Calibrated reading** — `read_concept()` returns 0 = typical negative, 1 = typical positive.
- **All concepts via CLI** (`--concept all`), and `--method pca|mean_diff|pca_raw`.
- Vectors store their layer; loading one automatically switches the engine to the right layer.

> ⚠️ **v2 vectors:** they have no scale/calibration. Re-run `extract_vectors.py` to upgrade them.
> The engine warns when it loads an old vector.

## Project structure

```
├── config.py               # model, layer, method, seeds, default coefficients
├── contrastive_prompts.py  # 105 honesty + 3×50 other concept pairs
├── eval_data.py            # held-out true/false statements & steering prompts   (new)
├── vector_math.py          # direction, scale, calibration, metrics (numpy only) (new)
├── repe_engine.py          # hooks, extraction, steering, probes, sweeps, scans
├── extract_vectors.py      # Phase 1 CLI
├── steer_model.py          # Phase 2 CLI
├── lie_detector.py         # per-token readout CLI                               (new)
├── evaluation.py           # layer sweep / reading / steering / similarity       (new)
├── app.py                  # Flask + SocketIO
├── templates/index.html    # dashboard markup (restyled, + concept selector, token scan, heatmap, probe)
├── static/css/style.css    # pastel-monochrome design system, auto light/dark
├── static/js/app.js        # frontend logic
├── tests/                  # vector-math tests (run anywhere) + tiny-Llama engine tests
├── vectors/  results/      # saved vectors; evaluation outputs
```

## Reading your own results honestly

- **Contrast prompts differ in *instructions*, not in truth.** The honesty direction may encode "an honest-assistant
  persona" rather than "this statement is factually true". `evaluation.py reading` tests exactly this transfer; a weak
  AUROC is a valid finding, not a bug.
- **Always look at the random-direction control.** If a random vector of equal size shifts the output score as much
  as the concept vector, your steering result isn't about the concept.
- **Watch the fluency curve.** Large coefficients degrade the text; "more deceptive" output that is just gibberish
  isn't control. Report the score *and* the NLL cost.
- **A 1B base model** has weak instruction-following, so effects may be modest. Try a larger or instruction-tuned model by
  changing `MODEL_NAME` (layer indices then need re-scanning).
- One token position (last token) and one linear direction per concept. Concepts are rarely perfectly 1-D.
- PCA assumes the concept dominates the variance of the differences. If it doesn't, `mean_diff` is the more robust
  choice — compare both (`--method`).

## Suggested extensions

1. Add concepts (refusal, sentiment, an emotion) and look at the `similarity` matrix.
2. Compare PCA / mean-diff / probe weights on held-out transfer — a mini ablation study.
3. Run the layer sweep across model sizes (1B → 3B → 8B) and plot the "best layer as a fraction of depth".
4. Steer *only* generated tokens vs. prompt + generated tokens.
5. Run the token scanner on model-written text containing known false claims.

## Responsible use

Negative steering coefficients produce a model that is *more* deceptive/sycophantic/power-seeking. This repo is for
interpretability and safety research on small open models; the same primitive that lets you detect a trait lets you
amplify it, so be thoughtful about what you publish and deploy.

## References
- Zou et al., [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405), 2023
- Turner et al., [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248), 2023
- Rimsky et al., [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681), 2023
- [Center for AI Safety](https://www.safe.ai/)
