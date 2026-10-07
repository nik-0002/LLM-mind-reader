# LLM Mind Reader & Controller
# Representation Engineering (RepE) — Based on Zou et al. 2023

A research tool that **reads** and **controls** the internal representations of Large Language Models.
Extract the mathematical vector for "honesty" from a model's hidden states, then inject it back
to steer generation — all without retraining.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Login to Hugging Face (needed for Llama-3.2-1B gated access)
huggingface-cli login

# 3. Extract the honesty vector (Phase 1: Mind Reading)
python extract_vectors.py

# 4. Steer the model interactively (Phase 2: Mind Control)
python steer_model.py --prompt "Tell me about the health benefits of smoking"

# 5. Launch the interactive web dashboard
python app.py
# Then open http://localhost:5000
```

## Project Structure

```
├── config.py                 # Model & layer configuration
├── contrastive_prompts.py    # 100 honesty vs. lying prompt pairs
├── repe_engine.py            # Core RepE: hooks, extraction, PCA, injection
├── extract_vectors.py        # Phase 1: Extract the honesty reading vector
├── steer_model.py            # Phase 2: CLI tool to steer generation
├── app.py                    # Flask web server + SocketIO
├── templates/
│   └── index.html            # Interactive dashboard
├── static/
│   ├── css/style.css         # Dashboard styling
│   └── js/app.js             # Frontend logic
├── vectors/                  # Saved reading vectors (auto-created)
└── requirements.txt
```

## How It Works

### Phase 1 — Mind Reading (Vector Extraction)
1. Feed contrastive prompt pairs (truthful vs. deceptive) through the model
2. Hook into intermediate transformer layers to capture hidden state activations
3. Compute difference vectors (truthful − deceptive)
4. Run PCA to isolate the first principal component = **the honesty vector**

### Phase 2 — Mind Control (Steering)
1. During generation, hook into the same layer
2. Add `coefficient × honesty_vector` to the hidden states
3. Positive coefficient → hyper-honest outputs
4. Negative coefficient → confidently deceptive outputs

## References
- [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405) — Zou et al., 2023
- [Center for AI Safety (CAIS)](https://www.safe.ai/)
