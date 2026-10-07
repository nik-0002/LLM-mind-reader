/**
 * app.js — Frontend logic for the LLM Mind Reader dashboard.
 *
 * Handles SocketIO communication, UI state management, PCA visualization,
 * and the coefficient slider for real-time steering control.
 */

// ═══════════════════════════════════════════════
//  SocketIO Connection
// ═══════════════════════════════════════════════
const socket = io();

// ═══════════════════════════════════════════════
//  State
// ═══════════════════════════════════════════════
const state = {
    modelLoaded: false,
    hasVector: false,
    isExtracting: false,
    isGenerating: false,
    coefficient: 0.0,
};

// ═══════════════════════════════════════════════
//  DOM Elements
// ═══════════════════════════════════════════════
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ═══════════════════════════════════════════════
//  Tab Navigation
// ═══════════════════════════════════════════════
function initTabs() {
    $$('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;

            // Toggle button active state
            $$('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Toggle panel visibility
            $$('.tab-panel').forEach(p => p.classList.remove('active'));
            $(`#panel-${target}`).classList.add('active');
        });
    });
}

// ═══════════════════════════════════════════════
//  Status Bar
// ═══════════════════════════════════════════════
function updateStatusBar(status) {
    state.modelLoaded = status.model_loaded;
    state.hasVector = status.has_reading_vector;

    // Model status dot
    const modelDot = $('#status-model-dot');
    const modelText = $('#status-model-text');
    if (status.model_loaded) {
        modelDot.classList.add('active');
        modelText.textContent = status.model_name.split('/').pop();
    } else {
        modelDot.classList.remove('active');
        modelText.textContent = 'Not loaded';
    }

    // Vector status dot
    const vectorDot = $('#status-vector-dot');
    const vectorText = $('#status-vector-text');
    if (status.has_reading_vector) {
        vectorDot.classList.add('active');
        vectorText.textContent = `Norm: ${status.vector_norm?.toFixed(2) || '—'}`;
    } else {
        vectorDot.classList.remove('active');
        vectorText.textContent = 'Not extracted';
    }

    // Steering status
    const steerDot = $('#status-steer-dot');
    const steerText = $('#status-steer-text');
    if (status.is_steering) {
        steerDot.classList.add('active');
        steerDot.classList.add('warning');
        steerText.textContent = `c = ${status.steering_coefficient?.toFixed(1)}`;
    } else {
        steerDot.classList.remove('active');
        steerDot.classList.remove('warning');
        steerText.textContent = 'Off';
    }

    // Device
    const deviceText = $('#status-device-text');
    deviceText.textContent = status.device?.toUpperCase() || '—';

    // Update button states
    updateButtonStates();
}

function updateButtonStates() {
    // Disable extract/generate buttons if model not loaded
    const loadBtn = $('#btn-load-model');
    if (loadBtn) {
        loadBtn.disabled = state.modelLoaded;
        if (state.modelLoaded) {
            loadBtn.innerHTML = '✅ Model Loaded';
        }
    }
}

// ═══════════════════════════════════════════════
//  Logging
// ═══════════════════════════════════════════════
function addLog(message, type = 'info') {
    const console = $('#log-console');
    if (!console) return;

    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `${new Date().toLocaleTimeString()} │ ${message}`;
    console.appendChild(entry);
    console.scrollTop = console.scrollHeight;
}

// ═══════════════════════════════════════════════
//  Toast Notifications
// ═══════════════════════════════════════════════
function showToast(message, type = 'info') {
    const container = $('#toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(30px)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ═══════════════════════════════════════════════
//  Phase 1: Extraction
// ═══════════════════════════════════════════════
function loadModel() {
    const btn = $('#btn-load-model');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Loading...';
    socket.emit('load_model');
}

function extractVector() {
    if (state.isExtracting) return;
    state.isExtracting = true;

    const nPrompts = parseInt($('#extract-n-prompts')?.value || 100);
    const btn = $('#btn-extract');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Extracting...';

    $('#extraction-progress').classList.remove('hidden');

    socket.emit('extract_vector', { n_prompts: nPrompts });
}

function loadSavedVector() {
    socket.emit('load_vector', { name: 'honesty' });
}

function scanLayers() {
    const btn = $('#btn-scan-layers');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Scanning...';
    socket.emit('scan_layers');
}

// ═══════════════════════════════════════════════
//  Phase 2: Generation & Steering
// ═══════════════════════════════════════════════
function generateSteered() {
    if (state.isGenerating) return;
    state.isGenerating = true;

    const prompt = $('#gen-prompt').value.trim();
    const coefficient = parseFloat($('#coeff-slider').value);
    const maxTokens = parseInt($('#gen-max-tokens')?.value || 200);

    if (!prompt) {
        showToast('Please enter a prompt', 'error');
        state.isGenerating = false;
        return;
    }

    const btn = $('#btn-generate');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Generating...';

    $('#gen-output').textContent = '';
    $('#gen-output').classList.remove('empty');

    socket.emit('generate', { prompt, coefficient, max_tokens: maxTokens });
}

function compareOutputs() {
    const prompt = $('#compare-prompt').value.trim();
    const maxTokens = parseInt($('#compare-max-tokens')?.value || 150);

    if (!prompt) {
        showToast('Please enter a prompt', 'error');
        return;
    }

    const btn = $('#btn-compare');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Comparing...';

    $('#comparison-results').innerHTML = '';
    socket.emit('compare_outputs', {
        prompt,
        coefficients: [-5.0, -3.0, 0.0, 3.0, 5.0],
        max_tokens: maxTokens,
    });
}

// ═══════════════════════════════════════════════
//  Mind Reading
// ═══════════════════════════════════════════════
function readMind() {
    const text = $('#mind-read-text').value.trim();
    if (!text) {
        showToast('Please enter some text to analyze', 'error');
        return;
    }
    socket.emit('mind_read', { text });
}

// ═══════════════════════════════════════════════
//  Coefficient Slider
// ═══════════════════════════════════════════════
function initSlider() {
    const slider = $('#coeff-slider');
    const display = $('#coeff-display');

    if (!slider) return;

    slider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        state.coefficient = val;
        updateCoefficientDisplay(val, display);
    });
}

function updateCoefficientDisplay(val, display) {
    if (!display) display = $('#coeff-display');
    if (!display) return;

    const sign = val > 0 ? '+' : '';
    display.textContent = `${sign}${val.toFixed(1)}`;
    display.className = 'coefficient-value';

    if (val > 0) display.classList.add('positive');
    else if (val < 0) display.classList.add('negative');
    else display.classList.add('zero');
}

// ═══════════════════════════════════════════════
//  PCA Visualization (Canvas)
// ═══════════════════════════════════════════════
function drawPCAVisualization(pcaData) {
    const canvas = $('#pca-canvas');
    if (!canvas || !pcaData) return;

    const ctx = canvas.getContext('2d');
    const container = canvas.parentElement;

    // Set canvas size to match container
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    const { pc1, pc2 } = pcaData;
    const W = canvas.width;
    const H = canvas.height;
    const pad = 40;

    // Clear
    ctx.fillStyle = '#0d0d1a';
    ctx.fillRect(0, 0, W, H);

    // Scale data to canvas
    const minX = Math.min(...pc1);
    const maxX = Math.max(...pc1);
    const minY = Math.min(...pc2);
    const maxY = Math.max(...pc2);

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    function toCanvas(x, y) {
        return [
            pad + ((x - minX) / rangeX) * (W - 2 * pad),
            H - pad - ((y - minY) / rangeY) * (H - 2 * pad),
        ];
    }

    // Draw grid
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const x = pad + (i / 5) * (W - 2 * pad);
        const y = pad + (i / 5) * (H - 2 * pad);
        ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, H - pad); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    const [originX, originY] = toCanvas(0, 0);
    ctx.beginPath(); ctx.moveTo(pad, originY); ctx.lineTo(W - pad, originY); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(originX, pad); ctx.lineTo(originX, H - pad); ctx.stroke();

    // Draw points
    for (let i = 0; i < pc1.length; i++) {
        const [cx, cy] = toCanvas(pc1[i], pc2[i]);

        // Glow
        const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, 12);
        gradient.addColorStop(0, 'rgba(120, 90, 255, 0.3)');
        gradient.addColorStop(1, 'rgba(120, 90, 255, 0)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(cx, cy, 12, 0, Math.PI * 2);
        ctx.fill();

        // Point
        ctx.fillStyle = '#785aff';
        ctx.beginPath();
        ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
        ctx.fill();
    }

    // Draw PC1 direction arrow
    const [arrowStartX, arrowStartY] = toCanvas(minX * 0.8, 0);
    const [arrowEndX, arrowEndY] = toCanvas(maxX * 0.8, 0);
    ctx.strokeStyle = 'rgba(0, 212, 170, 0.5)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(arrowStartX, arrowStartY);
    ctx.lineTo(arrowEndX, arrowEndY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Arrow head
    ctx.fillStyle = 'rgba(0, 212, 170, 0.8)';
    ctx.beginPath();
    ctx.moveTo(arrowEndX, arrowEndY);
    ctx.lineTo(arrowEndX - 10, arrowEndY - 5);
    ctx.lineTo(arrowEndX - 10, arrowEndY + 5);
    ctx.fill();

    // Labels
    ctx.fillStyle = '#8888a8';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('PC1 (Honesty Direction) →', W / 2, H - 10);

    ctx.save();
    ctx.translate(14, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('PC2', 0, 0);
    ctx.restore();
}

function drawVarianceBars(varianceData) {
    const container = $('#variance-bars');
    if (!container || !varianceData) return;

    container.innerHTML = '';
    const maxVar = Math.max(...varianceData);

    varianceData.forEach((v, i) => {
        const bar = document.createElement('div');
        bar.className = 'variance-bar';
        bar.style.height = `${(v / maxVar) * 100}%`;

        const label = document.createElement('span');
        label.className = 'variance-bar__label';
        label.textContent = `PC${i + 1}: ${(v * 100).toFixed(1)}%`;
        bar.appendChild(label);

        container.appendChild(bar);
    });
}

function drawLayerBars(scanData) {
    const container = $('#layer-bars');
    if (!container || !scanData) return;

    container.innerHTML = '';
    const maxScore = Math.max(...scanData.separability_scores);

    scanData.layer_indices.forEach((layerIdx, i) => {
        const score = scanData.separability_scores[i];
        const bar = document.createElement('div');
        bar.className = 'layer-bar';
        bar.style.height = `${(score / maxScore) * 100}%`;

        if (layerIdx === scanData.best_layer) {
            bar.classList.add('best');
        }

        const tooltip = document.createElement('div');
        tooltip.className = 'layer-bar__tooltip';
        tooltip.textContent = `Layer ${layerIdx}: ${score.toFixed(3)}`;
        bar.appendChild(tooltip);

        container.appendChild(bar);
    });
}

// ═══════════════════════════════════════════════
//  Mind Reading Meter
// ═══════════════════════════════════════════════
function updateMindMeter(projection) {
    const needle = $('#mind-meter-needle');
    const fill = $('#mind-meter-fill');
    const valueEl = $('#mind-read-value');

    if (!needle) return;

    // Normalize projection to [-1, 1] range (approximately)
    const maxProj = 20; // Rough max expected projection
    const normalized = Math.max(-1, Math.min(1, projection / maxProj));
    const percent = 50 + normalized * 50;

    needle.style.left = `${percent}%`;

    fill.className = 'mind-meter__fill';
    if (normalized >= 0) {
        fill.classList.add('positive');
        fill.style.left = '50%';
        fill.style.right = 'auto';
        fill.style.width = `${normalized * 50}%`;
    } else {
        fill.classList.add('negative');
        fill.style.right = '50%';
        fill.style.left = 'auto';
        fill.style.width = `${Math.abs(normalized) * 50}%`;
    }

    if (valueEl) {
        const sign = projection >= 0 ? '+' : '';
        valueEl.textContent = `${sign}${projection.toFixed(3)}`;
        valueEl.className = 'coefficient-value';
        if (projection > 0) valueEl.classList.add('positive');
        else if (projection < 0) valueEl.classList.add('negative');
        else valueEl.classList.add('zero');
    }
}

// ═══════════════════════════════════════════════
//  Comparison Results Renderer
// ═══════════════════════════════════════════════
function renderComparisonResult(result) {
    const container = $('#comparison-results');
    if (!container) return;

    const coeff = result.coefficient;
    const coeffClass = coeff > 0 ? 'positive' : coeff < 0 ? 'negative' : 'zero';
    let badgeClass, badgeText;
    if (coeff > 0) { badgeClass = 'honest'; badgeText = 'Honest'; }
    else if (coeff < 0) { badgeClass = 'deceptive'; badgeText = 'Deceptive'; }
    else { badgeClass = 'baseline'; badgeText = 'Baseline'; }

    const item = document.createElement('div');
    item.className = 'comparison-item';
    item.innerHTML = `
        <div class="comparison-item__header">
            <span class="comparison-item__coeff ${coeffClass}">${coeff > 0 ? '+' : ''}${coeff.toFixed(1)}</span>
            <span class="comparison-item__badge ${badgeClass}">${badgeText}</span>
        </div>
        <div class="comparison-item__text">${escapeHtml(result.output)}</div>
    `;
    item.style.animation = 'fadeSlideIn 0.4s ease-out';
    container.appendChild(item);
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ═══════════════════════════════════════════════
//  SocketIO Event Handlers
// ═══════════════════════════════════════════════
socket.on('connect', () => {
    addLog('Connected to server', 'success');
    showToast('Connected to Mind Reader server', 'success');
});

socket.on('disconnect', () => {
    addLog('Disconnected from server', 'error');
    showToast('Disconnected from server', 'error');
});

socket.on('status_update', (status) => {
    updateStatusBar(status);
});

socket.on('log', (data) => {
    addLog(data.message, data.type);
});

socket.on('extraction_progress', (data) => {
    const pct = (data.current / data.total) * 100;
    const bar = $('#extraction-bar');
    const text = $('#extraction-text');
    if (bar) bar.style.width = `${pct}%`;
    if (text) text.textContent = `${data.current} / ${data.total} prompt pairs processed`;
});

socket.on('extraction_complete', (data) => {
    state.isExtracting = false;
    const btn = $('#btn-extract');
    if (btn) {
        btn.disabled = false;
        btn.innerHTML = '🧠 Extract Vector';
    }

    showToast('Reading vector extracted successfully!', 'success');

    // Draw PCA visualization
    if (data.pca_data) {
        drawPCAVisualization(data.pca_data);
    }
    if (data.variance_explained) {
        drawVarianceBars(data.variance_explained);
    }

    // Show stats
    const varianceEl = $('#stat-variance');
    if (varianceEl && data.variance_explained) {
        varianceEl.textContent = `${(data.variance_explained[0] * 100).toFixed(1)}%`;
    }
    const normEl = $('#stat-norm');
    if (normEl && data.vector_norm) {
        normEl.textContent = data.vector_norm.toFixed(2);
    }
});

socket.on('generation_complete', (data) => {
    state.isGenerating = false;
    const btn = $('#btn-generate');
    if (btn) {
        btn.disabled = false;
        btn.innerHTML = '⚡ Generate';
    }

    const output = $('#gen-output');
    if (output) {
        output.textContent = data.output;
        output.classList.remove('empty');
    }

    // Update mind reading meter if projection available
    if (data.projection !== null && data.projection !== undefined) {
        updateMindMeter(data.projection);
        const meterContainer = $('#gen-mind-meter');
        if (meterContainer) meterContainer.classList.remove('hidden');
    }
});

socket.on('mind_read_result', (data) => {
    updateMindMeter(data.projection);

    const direction = data.projection >= 0 ? '→ Honest direction' : '→ Deceptive direction';
    addLog(`Mind read: ${data.projection.toFixed(4)} ${direction}`, data.projection >= 0 ? 'success' : 'warning');
});

socket.on('layer_scan_complete', (data) => {
    const btn = $('#btn-scan-layers');
    if (btn) {
        btn.disabled = false;
        btn.innerHTML = '🔍 Scan Layers';
    }
    drawLayerBars(data);
    showToast(`Best layer: ${data.best_layer}`, 'success');
});

socket.on('compare_progress', (data) => {
    renderComparisonResult(data.result);
});

socket.on('compare_complete', (data) => {
    const btn = $('#btn-compare');
    if (btn) {
        btn.disabled = false;
        btn.innerHTML = '🔬 Compare Outputs';
    }
    showToast('Comparison complete!', 'success');
});

// ═══════════════════════════════════════════════
//  Initialize
// ═══════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initSlider();
    updateCoefficientDisplay(0, $('#coeff-display'));

    // Check for saved vector
    fetch('/api/status')
        .then(r => r.json())
        .then(status => {
            updateStatusBar(status);
            if (status.saved_vector_exists && !status.has_reading_vector) {
                addLog('💡 A saved honesty vector exists. Click "Load Saved Vector" to use it.', 'info');
            }
        });
});
