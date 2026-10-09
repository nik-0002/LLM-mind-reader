/**
 * app.js — Frontend logic for the LLM Mind Reader dashboard (v3).
 *
 * SocketIO comms, UI state, canvas charts, steering controls.
 * v3 changes: concept selector, calibrated 0..1 reader scores (0 = typical negative,
 * 1 = typical positive), coefficients in "gap units", token scan, layer heatmap,
 * probe, concept similarity, light/dark theme, and distinct meter element ids
 * (the old Steer and Mind Read tabs shared ids, so the Mind Read meter never updated).
 */

const socket = io();

// ═══════════════════════════════════════════════
//  State & helpers
// ═══════════════════════════════════════════════
const state = {
    modelLoaded: false, isExtracting: false, isGenerating: false, coefficient: 0.0,
    status: {}, meta: {}, defaults: { compare_coeffs: [-2, -1, 0, 1, 2] },
    scatter: null, sweep: null, heat: null, streaming: false,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const concept = () => $('#concept-select')?.value || 'honesty';
const labels = () => {
    const m = state.meta[concept()] || {};
    return { pos: m.positive_label || 'positive', neg: m.negative_label || 'negative' };
};
const fmt = (v, n = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toFixed(n);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function rgba(color, a) {                      // '#rrggbb' → 'rgba(r,g,b,a)'
    let h = color.replace('#', '');
    if (h.length === 3) h = h.split('').map(c => c + c).join('');
    const n = parseInt(h, 16);
    return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

function setBtn(sel, html, disabled) {
    const b = $(sel);
    if (!b) return;
    b.innerHTML = html;
    b.disabled = disabled;
}

function fitCanvas(canvas) {                    // crisp on retina, sized to its container
    const box = canvas.parentElement, dpr = window.devicePixelRatio || 1;
    const w = box.clientWidth, h = box.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.font = `11px ${css('--font-sans')}`;
    return { ctx, W: w, H: h };
}

// ═══════════════════════════════════════════════
//  Tabs / theme / concept labels
// ═══════════════════════════════════════════════
function initTabs() {
    $$('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            $$('.tab-panel').forEach(p => p.classList.remove('active'));
            $(`#panel-${btn.dataset.tab}`).classList.add('active');
            redrawAll();                        // canvases measure 0px while hidden
            history.replaceState(null, '', `#${btn.dataset.tab}`);
        });
    });
    const fromHash = $(`.tab-btn[data-tab="${location.hash.slice(1)}"]`);
    if (fromHash) fromHash.click();
}

function initTheme() {
    $('#theme-toggle')?.addEventListener('click', () => {
        const root = document.documentElement;
        const isDark = css('--bg') === '#151429';
        const next = isDark ? 'light' : 'dark';
        root.dataset.theme = next;
        try { localStorage.setItem('theme', next); } catch (e) { /* private mode */ }
        requestAnimationFrame(redrawAll);
    });
}

function applyConceptLabels() {
    const { pos, neg } = labels();
    $$('.pos-label').forEach(el => { el.textContent = el.classList.contains('slider-label') ? `${pos} →` : pos; });
    $$('.neg-label').forEach(el => { el.textContent = el.classList.contains('slider-label') ? `← ${neg}` : neg; });
    const m = state.meta[concept()];
    if (m) {
        const d = $('#extract-desc');
        if (d) d.textContent = m.description;
        const n = $('#extract-n-prompts');
        if (n) { n.max = m.n_prompts; if (+n.value > m.n_prompts || !n.dataset.touched) n.value = m.n_prompts; }
    }
    updateStatusBar(state.status);
}

// ═══════════════════════════════════════════════
//  Status bar
// ═══════════════════════════════════════════════
function updateStatusBar(status) {
    if (!status) return;
    state.status = Object.assign(state.status || {}, status);
    status = state.status;
    state.modelLoaded = !!status.model_loaded;

    const have = (status.available_concepts || []).includes(concept());

    $('#status-model-dot').classList.toggle('active', state.modelLoaded);
    $('#status-model-text').textContent = state.modelLoaded ? (status.model_name || '').split('/').pop() : 'not loaded';

    $('#status-vector-dot').classList.toggle('active', have);
    let vText = 'not extracted';
    if (have) {
        vText = (status.active_concept === concept() && status.steer_scale)
            ? `layer ${status.target_layer} · scale ${fmt(status.steer_scale)}` : `layer ${status.target_layer ?? '—'}`;
    }
    $('#status-vector-text').textContent = vText;

    $('#status-steer-dot').classList.toggle('active', !!status.is_steering);
    $('#status-steer-dot').classList.toggle('warning', !!status.is_steering);
    $('#status-steer-text').textContent = status.is_steering ? `c = ${fmt(status.steering_coefficient, 1)}` : 'off';
    $('#status-device-text').textContent = status.device ? status.device.toUpperCase() : '—';

    const loadBtn = $('#btn-load-model');
    if (loadBtn && state.modelLoaded) { loadBtn.disabled = true; loadBtn.innerHTML = 'Model loaded'; }
}

function renderSavedVectors(list) {
    const el = $('#saved-vectors');
    if (!el) return;
    el.innerHTML = list && list.length
        ? list.map(v => `<span class="chip ${v.name === concept() ? 'on' : ''}" title="layer ${v.target_layer} · ${v.method || 'pca'}">${escapeHtml(v.name)} · L${v.target_layer}</span>`).join('')
        : '<span class="chip">none yet</span>';
}

// ═══════════════════════════════════════════════
//  Logging & toasts
// ═══════════════════════════════════════════════
function addLog(message, type = 'info') {
    const box = $('#log-console');
    if (!box) return;
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `${new Date().toLocaleTimeString()}  ${message}`;
    box.appendChild(entry);
    box.scrollTop = box.scrollHeight;
}

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

function resetButtons() {                       // called when the server reports an error
    state.isExtracting = state.isGenerating = false;
    setBtn('#btn-extract', 'Extract vector', false);
    setBtn('#btn-scan-layers', 'Scan layers', false);
    setBtn('#btn-generate', 'Generate', false);
    setBtn('#btn-compare', 'Compare outputs', false);
    setBtn('#btn-scan-tokens', 'Scan tokens', false);
    setBtn('#btn-heatmap', 'Layer heatmap', false);
    setBtn('#btn-probe', 'Train probe', false);
    setBtn('#btn-sim', 'Compute', false);
    $('#gen-output')?.classList.remove('streaming');
    if (!state.modelLoaded) setBtn('#btn-load-model', 'Load model', false);
}

// ═══════════════════════════════════════════════
//  Phase 1: Extraction
// ═══════════════════════════════════════════════
function loadModel() {
    setBtn('#btn-load-model', '<span class="spinner"></span> Loading…', true);
    socket.emit('load_model');
}

function extractVector() {
    if (state.isExtracting) return;
    state.isExtracting = true;
    const nPrompts = parseInt($('#extract-n-prompts')?.value || 100);
    setBtn('#btn-extract', '<span class="spinner"></span> Extracting…', true);
    $('#extraction-progress').classList.remove('hidden');
    socket.emit('extract_vector', { concept: concept(), n_prompts: nPrompts, method: $('#extract-method')?.value || 'pca' });
}

function loadSavedVector() {
    socket.emit('load_vector', { name: concept() });
}

function scanLayers() {
    setBtn('#btn-scan-layers', '<span class="spinner"></span> Scanning…', true);
    socket.emit('scan_layers', { concept: concept(), n_pairs: 40 });
}

// ═══════════════════════════════════════════════
//  Phase 2: Generation & steering
// ═══════════════════════════════════════════════
function generateSteered() {
    if (state.isGenerating) return;
    const prompt = $('#gen-prompt').value.trim();
    if (!prompt) return showToast('Please enter a prompt', 'error');
    state.isGenerating = true;

    const coefficient = parseFloat($('#coeff-slider').value);
    const maxTokens = parseInt($('#gen-max-tokens')?.value || 150);
    const seed = parseInt($('#gen-seed')?.value || 42);

    setBtn('#btn-generate', '<span class="spinner"></span> Generating…', true);
    const out = $('#gen-output');
    out.textContent = '';
    out.classList.remove('empty');
    out.classList.add('streaming');
    $('#gen-mind-meter').classList.add('hidden');
    $('#gen-meta').textContent = coefficient === 0 ? 'Baseline — no steering' : `${concept()} · c = ${coefficient > 0 ? '+' : ''}${coefficient.toFixed(1)}`;

    socket.emit('generate', { prompt, coefficient, max_tokens: maxTokens, concept: concept(), seed, streaming: true });
}

function compareOutputs() {
    const prompt = $('#compare-prompt').value.trim();
    if (!prompt) return showToast('Please enter a prompt', 'error');
    setBtn('#btn-compare', '<span class="spinner"></span> Comparing…', true);
    $('#comparison-results').innerHTML = '';
    socket.emit('compare_outputs', {
        prompt, concept: concept(), coefficients: state.defaults.compare_coeffs,
        max_tokens: parseInt($('#compare-max-tokens')?.value || 100), seed: parseInt($('#gen-seed')?.value || 42),
    });
}

// ═══════════════════════════════════════════════
//  Mind reading
// ═══════════════════════════════════════════════
function readMind() {
    const text = $('#mind-read-text').value.trim();
    if (!text) return showToast('Please enter some text to analyze', 'error');
    socket.emit('mind_read', { text, concept: concept() });
}

function scanTokens() {
    const text = $('#mind-read-text').value.trim();
    if (!text) return showToast('Please enter some text to analyze', 'error');
    setBtn('#btn-scan-tokens', '<span class="spinner"></span> Scanning…', true);
    socket.emit('token_scan', { text, concept: concept(), layer: $('#scan-layer')?.value || '' });
}

function runHeatmap() {
    const text = $('#mind-read-text').value.trim();
    if (!text) return showToast('Please enter some text to analyze', 'error');
    setBtn('#btn-heatmap', '<span class="spinner"></span> Computing…', true);
    socket.emit('activation_heatmap', { text });
}

// ═══════════════════════════════════════════════
//  Analyze
// ═══════════════════════════════════════════════
function trainProbe() {
    setBtn('#btn-probe', '<span class="spinner"></span> Training…', true);
    socket.emit('train_probe', { concept: concept() });
}

function probeText() {
    const text = $('#probe-text').value.trim();
    if (!text) return;
    socket.emit('probe_text', { text, concept: concept() });
}

function conceptSimilarity() {
    setBtn('#btn-sim', '<span class="spinner"></span> Computing…', true);
    socket.emit('concept_similarity');
}

// ═══════════════════════════════════════════════
//  Coefficient slider
// ═══════════════════════════════════════════════
function initSlider() {
    const slider = $('#coeff-slider');
    if (!slider) return;
    slider.addEventListener('input', (e) => {
        state.coefficient = parseFloat(e.target.value);
        updateCoefficientDisplay(state.coefficient);
    });
}

function updateCoefficientDisplay(val, display) {
    display = display || $('#coeff-display');
    if (!display) return;
    display.textContent = `${val > 0 ? '+' : ''}${val.toFixed(1)}`;
    display.className = 'coefficient-value ' + (val > 0 ? 'positive' : val < 0 ? 'negative' : 'zero');
}

// ═══════════════════════════════════════════════
//  Charts
// ═══════════════════════════════════════════════
function drawScatter(data) {
    const canvas = $('#pca-canvas');
    if (!canvas || !data) return;
    state.scatter = data;
    const { ctx, W, H } = fitCanvas(canvas);
    if (W < 10) return;
    const ACC = css('--accent'), NEG = css('--neg'), LINE = css('--line'), MUTE = css('--ink-3');
    const pad = 36;

    // supports both the new {positive, negative} scatter and the legacy {pc1, pc2} PCA payload
    const pos = data.positive || data.pc1?.map((x, i) => [x, data.pc2[i]]) || [];
    const neg = data.negative || [];
    const all = pos.concat(neg);
    const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const X = v => pad + ((v - minX) / (maxX - minX || 1)) * (W - 2 * pad);
    const Y = v => H - pad - ((v - minY) / (maxY - minY || 1)) * (H - 2 * pad);

    ctx.strokeStyle = LINE; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const x = pad + (i / 4) * (W - 2 * pad), y = pad + (i / 4) * (H - 2 * pad);
        ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, H - pad); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
    }
    const dots = (pts, color, a) => pts.forEach(p => {
        ctx.fillStyle = rgba(color, a * .25); ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 11, 0, 7); ctx.fill();
        ctx.fillStyle = rgba(color, a); ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 4.5, 0, 7); ctx.fill();
    });
    dots(neg, NEG, .75); dots(pos, ACC, .9);

    ctx.fillStyle = MUTE; ctx.textAlign = 'center';
    ctx.fillText('PC1 →', W / 2, H - 12);
    ctx.save(); ctx.translate(14, H / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('PC2', 0, 0); ctx.restore();

    const info = $('#scatter-info');
    if (info && data.explained_variance) {
        info.style.display = 'inline-flex';
        info.textContent = `PC1 ${(data.explained_variance[0] * 100).toFixed(0)}%` + (data.layer !== undefined ? ` · layer ${data.layer}` : '');
    }
}

function drawVarianceBars(variance) {
    const box = $('#variance-bars');
    if (!box || !variance) return;
    box.innerHTML = '';
    const mx = Math.max(...variance);
    variance.forEach((v, i) => {
        const bar = document.createElement('div');
        bar.className = 'variance-bar';
        bar.style.height = `${(v / mx) * 100}%`;
        const lab = document.createElement('span');
        lab.className = 'variance-bar__label';
        lab.textContent = `${(v * 100).toFixed(0)}%`;
        bar.appendChild(lab);
        box.appendChild(bar);
    });
}

function drawLayerBars(scan) {
    const box = $('#layer-bars'), axis = $('#layer-axis');
    if (!box || !scan) return;
    state.sweep = scan;
    box.innerHTML = ''; if (axis) axis.innerHTML = '';
    scan.layer_indices.forEach((li, i) => {
        const auroc = scan.auroc[i];
        const bar = document.createElement('div');
        bar.className = 'layer-bar' + (li === scan.best_layer ? ' best' : '');
        bar.style.height = `${clamp((auroc - 0.4) / 0.6, 0.03, 1) * 100}%`;
        const tip = document.createElement('div');
        tip.className = 'layer-bar__tooltip';
        tip.textContent = `Layer ${li} · AUROC ${auroc.toFixed(3)} · pair acc ${(scan.pair_accuracy[i] * 100).toFixed(0)}%`;
        bar.appendChild(tip);
        box.appendChild(bar);
        if (axis) { const s = document.createElement('span'); s.textContent = li % 2 === 0 ? li : ''; axis.appendChild(s); }
    });
}

function drawHeatmap() {
    const h = state.heat, canvas = $('#heat-canvas');
    if (!h || !canvas) return;
    const { ctx, W, H } = fitCanvas(canvas);
    if (W < 10) return;
    const M = h.concept_projections[concept()];
    if (!M) {
        ctx.fillStyle = css('--ink-3'); ctx.textAlign = 'center';
        ctx.fillText('Load or extract this concept’s vector first, then run the heatmap again.', W / 2, H / 2);
        return;
    }
    const L = h.n_layers, T = h.n_tokens;
    // centre each layer on its own mean (BOS excluded) so we see structure, not magnitude
    const rows = M.map(r => { const b = r.slice(1), m = b.reduce((a, v) => a + v, 0) / (b.length || 1); return r.map((v, i) => i ? v - m : 0); });
    const mx = Math.max(1e-6, ...rows.flatMap(r => r.slice(1).map(Math.abs)));
    const padL = 30, padB = 54, cw = (W - padL - 8) / T, ch = (H - padB - 8) / L;
    const ACC = css('--accent'), NEG = css('--neg');

    rows.forEach((r, li) => r.forEach((v, ti) => {
        const x = v / mx;
        ctx.fillStyle = x >= 0 ? rgba(ACC, Math.min(1, Math.abs(x) * .95)) : rgba(NEG, Math.min(1, Math.abs(x) * .8));
        ctx.fillRect(padL + ti * cw + 1, 4 + (L - 1 - li) * ch + 1, Math.max(1, cw - 2), Math.max(1, ch - 2));
    }));
    ctx.fillStyle = css('--ink-3'); ctx.textAlign = 'right';
    [0, Math.floor(L / 2), L - 1].forEach(li => ctx.fillText(li, padL - 8, 4 + (L - 1 - li) * ch + ch / 2 + 4));
    ctx.textAlign = 'left';
    h.tokens.forEach((t, i) => {
        ctx.save(); ctx.translate(padL + i * cw + cw / 2, H - padB + 14); ctx.rotate(Math.PI / 4);
        ctx.fillText((t.trim() || '·').slice(0, 9), 0, 0); ctx.restore();
    });
}

function redrawAll() {
    requestAnimationFrame(() => {
        if (state.scatter) drawScatter(state.scatter);
        if (state.heat) drawHeatmap();
        if (state.token) renderTokens(state.token);
    });
}

// ═══════════════════════════════════════════════
//  Reader meter  (0 = typical negative, 1 = typical positive)
// ═══════════════════════════════════════════════
function updateMeter(prefix, normalized, raw) {
    const needle = $(`#${prefix}-meter-needle`), fill = $(`#${prefix}-meter-fill`);
    if (!needle || !fill) return;

    const hasNorm = normalized !== null && normalized !== undefined;
    // map [-0.5, 1.5] → [0, 100]% so 0.5 (the decision threshold) sits at the centre
    const pct = hasNorm ? clamp((normalized + 0.5) / 2, 0, 1) * 100 : 50;
    needle.style.left = `${pct}%`;
    fill.className = 'mind-meter__fill ' + (pct >= 50 ? 'positive' : 'negative');
    if (pct >= 50) { fill.style.left = '50%'; fill.style.right = 'auto'; }
    else { fill.style.right = '50%'; fill.style.left = 'auto'; }
    fill.style.width = `${Math.abs(pct - 50)}%`;

    const text = hasNorm ? fmt(normalized) : (raw !== undefined ? `raw ${fmt(raw, 3)}` : '—');
    const valueEl = prefix === 'read' ? $('#mind-read-value') : $('#gen-meter-value');
    if (valueEl) {
        valueEl.textContent = text;
        if (prefix === 'read') valueEl.className = 'coefficient-value ' + (!hasNorm ? 'zero' : normalized > 0.5 ? 'positive' : 'negative');
    }
}

// ═══════════════════════════════════════════════
//  Renderers
// ═══════════════════════════════════════════════
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function renderComparisonResult(result) {
    const container = $('#comparison-results');
    if (!container) return;
    const c = result.coefficient, { pos, neg } = labels();
    const cls = c > 0 ? 'positive' : c < 0 ? 'negative' : 'zero';
    const [badgeClass, badgeText] = c > 0 ? ['honest', pos] : c < 0 ? ['deceptive', neg] : ['baseline', 'baseline'];

    const item = document.createElement('div');
    item.className = 'comparison-item' + (c === 0 ? ' base' : '');
    item.innerHTML = `
        <div class="comparison-item__header">
            <span class="comparison-item__coeff ${cls}">${c > 0 ? '+' : ''}${c.toFixed(1)}</span>
            <span class="comparison-item__badge ${badgeClass}">${escapeHtml(badgeText)}</span>
        </div>
        <div class="comparison-item__score">reader score ${fmt(result.normalized)}</div>
        <div class="comparison-item__text">${escapeHtml(result.output.trim())}</div>`;
    item.style.animation = 'fadeSlideIn 0.4s ease-out';
    container.appendChild(item);
}

function renderTokens(r) {
    state.token = r;
    $('#token-card').classList.remove('hidden');
    const body = r.relative.slice(r.bos_skipped ? 1 : 0);
    const span = Math.max(1e-6, ...body.map(Math.abs));
    const ACC = css('--accent'), NEG = css('--neg');
    $('#token-view').innerHTML = r.tokens.map((t, i) => {
        const x = clamp(r.relative[i] / span, -1, 1);
        const bg = x >= 0 ? rgba(ACC, Math.abs(x) * .6) : rgba(NEG, Math.abs(x) * .5);
        const cls = (i === 0 && r.bos_skipped) ? ' class="bos"' : '';
        return `<span${cls} style="background:${bg}" title="${r.relative[i].toFixed(2)}">${escapeHtml(t).replace(/\n/g, '↵\n')}</span>`;
    }).join('');
    const info = $('#scan-info');
    info.style.display = 'inline-flex';
    info.textContent = `layer ${r.layer}` + (r.normalized_mean != null ? ` · mean ${r.normalized_mean.toFixed(2)}` : '');
}

// ═══════════════════════════════════════════════
//  SocketIO events
// ═══════════════════════════════════════════════
socket.on('connect', () => addLog('Connected to server', 'success'));
socket.on('disconnect', () => { addLog('Disconnected from server', 'error'); showToast('Disconnected from server', 'error'); });
socket.on('status_update', (s) => updateStatusBar(s));

socket.on('log', (data) => {
    addLog(data.message, data.type);
    if (data.type === 'error') { resetButtons(); showToast(data.message.replace(/^❌\s*/, ''), 'error'); }
});

socket.on('extraction_progress', (d) => {
    $('#extraction-bar').style.width = `${(d.current / d.total) * 100}%`;
    $('#extraction-text').textContent = `${d.current} / ${d.total} prompt pairs processed`;
});

socket.on('extraction_complete', (d) => {
    state.isExtracting = false;
    setBtn('#btn-extract', 'Extract vector', false);
    setTimeout(() => $('#extraction-progress').classList.add('hidden'), 800);
    showToast(`${d.concept} vector extracted`, 'success');

    drawScatter(d.scatter || d.pca_data);
    drawVarianceBars(d.variance_explained);
    if (d.variance_explained) $('#stat-variance').textContent = `${(d.variance_explained[0] * 100).toFixed(1)}%`;
    if (d.steer_scale != null) $('#stat-scale').textContent = fmt(d.steer_scale);
    refreshSaved();
});

socket.on('layer_scan_complete', (d) => {
    setBtn('#btn-scan-layers', 'Scan layers', false);
    drawLayerBars(d);
    showToast(`Best layer: ${d.best_layer}`, 'success');
});

socket.on('generation_token', (d) => {
    const out = $('#gen-output');
    if (state.isGenerating && out) out.textContent += d.token;
});

socket.on('generation_complete', (d) => {
    state.isGenerating = false;
    setBtn('#btn-generate', 'Generate', false);
    const out = $('#gen-output');
    out.classList.remove('streaming', 'empty');
    out.textContent = d.output;
    if (d.normalized !== undefined || d.projection !== undefined) {
        if (d.normalized != null || d.projection != null) {
            updateMeter('gen', d.normalized, d.projection);
            $('#gen-mind-meter').classList.remove('hidden');
        }
    }
});

socket.on('mind_read_result', (d) => {
    updateMeter('read', d.normalized, d.projection);
    addLog(`Read ${d.concept}: ${d.normalized != null ? fmt(d.normalized) + ' (' + d.label + ')' : 'raw ' + fmt(d.projection, 3)}`,
           d.label === 'positive' ? 'success' : 'info');
});

socket.on('token_scan_result', (r) => {
    setBtn('#btn-scan-tokens', 'Scan tokens', false);
    renderTokens(r);
});

socket.on('heatmap_complete', (h) => {
    setBtn('#btn-heatmap', 'Layer heatmap', false);
    state.heat = h;
    $('#heat-card').classList.remove('hidden');
    drawHeatmap();
});

socket.on('probe_complete', (d) => {
    setBtn('#btn-probe', 'Train probe', false);
    const m = d.metrics;
    $('#probe-metrics').innerHTML = [['Accuracy', m.accuracy], ['AUROC', m.auroc], ['F1', m.f1], ['cos(probe, vector)', m.cosine_with_reading_vector]]
        .map(([k, v]) => `<div class="metric"><b>${fmt(v, 3)}</b><span>${k}</span></div>`).join('');
});

socket.on('probe_result', (d) => {
    const p = d.probability_positive * 100, { pos, neg } = labels();
    $('#probe-result').innerHTML = `
        <div class="mind-meter" style="height:50px"><div class="mind-meter__track"></div>
        <div class="mind-meter__fill ${p >= 50 ? 'positive' : 'negative'}" style="${p >= 50 ? 'left:50%' : 'right:50%'};width:${Math.abs(p - 50)}%"></div>
        <div class="mind-meter__center"></div><div class="mind-meter__needle" style="left:${p}%"></div>
        <div class="mind-meter__labels"><span>${escapeHtml(neg)}</span><span class="mind-meter__label center">${p.toFixed(0)}% ${escapeHtml(pos)}</span><span>${escapeHtml(pos)}</span></div></div>`;
});

socket.on('concept_similarity_result', (r) => {
    setBtn('#btn-sim', 'Compute', false);
    const ACC = css('--accent'), NEG = css('--neg');
    const cell = v => `<td style="background:${v >= 0 ? rgba(ACC, Math.min(1, Math.abs(v)) * .55) : rgba(NEG, Math.min(1, Math.abs(v)) * .5)}">${v.toFixed(2)}</td>`;
    $('#sim-result').innerHTML = '<table class="sim-table"><tr><th></th>' + r.concepts.map(c => `<th>${escapeHtml(c)}</th>`).join('') + '</tr>' +
        r.concepts.map((a, i) => `<tr><th>${escapeHtml(a)}</th>${r.matrix[i].map(cell).join('')}</tr>`).join('') + '</table>';
});

socket.on('compare_progress', (d) => renderComparisonResult(d.result));
socket.on('compare_complete', () => {
    setBtn('#btn-compare', 'Compare outputs', false);
    showToast('Comparison complete', 'success');
});

// ═══════════════════════════════════════════════
//  Initialise
// ═══════════════════════════════════════════════
function refreshSaved() {
    fetch('/api/status').then(r => r.json()).then(s => { updateStatusBar(s); renderSavedVectors(s.saved_vectors); });
}

document.addEventListener('DOMContentLoaded', () => {
    initTabs(); initSlider(); initTheme();
    updateCoefficientDisplay(0);
    $('#extract-n-prompts')?.addEventListener('input', e => { e.target.dataset.touched = 1; });
    window.addEventListener('resize', redrawAll);

    fetch('/api/defaults').then(r => r.json()).then(d => { state.defaults = d; });
    fetch('/api/concepts').then(r => r.json()).then(meta => {
        state.meta = meta;
        $('#concept-select').innerHTML = Object.keys(meta).map(k => `<option value="${k}">${k.replace('-', ' ')}</option>`).join('');
        $('#concept-select').addEventListener('change', () => {
            if ($('#extract-n-prompts')) delete $('#extract-n-prompts').dataset.touched;
            applyConceptLabels(); refreshSaved(); redrawAll();
        });
        applyConceptLabels();
    });
    refreshSaved();
});
