/**
 * GridGuard SOC v2.0 — Dashboard Controller
 * Live telemetry plotting, attack simulation, diagnostics gallery,
 * animated KPI counters, residual histogram, and incident log.
 */

let isMlDefenseActive = true;

document.addEventListener('DOMContentLoaded', () => {
    initThemeToggle();
    initDefenseToggle();
    initLiveClock();
    initTelemetryChart();
    initResidualChart();
    initAttackSimulator();
    initDiagnosticsGallery();
    loadIncidentLog();
    loadAttackBreakdown();
    loadLiveMetrics();
    initTableSearch();
});

// ─── 1. Live Clock ────────────────────────────────────
function initLiveClock() {
    const el = document.getElementById('live-clock');
    const update = () => {
        const now = new Date();
        el.textContent = now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
    };
    update();
    setInterval(update, 1000);
}

// ─── 2. Live Metrics from API ──────────────────────────
async function loadLiveMetrics() {
    try {
        const res = await fetch('/api/metrics');
        if (!res.ok) return;
        const data = await res.json();

        const fp = data.forecasting_performance || {};
        const ids = data.cybersecurity_ids_performance || {};
        const atk = data.attack_type_breakdown || {};

        // Animate KPI values from API
        if (fp.r2_score)       animateValue('kpi-r2',        0, fp.r2_score * 100,      800, v => v.toFixed(2) + '%');
        if (fp.mae_mw)         animateValue('kpi-mae',        0, fp.mae_mw,              800, v => Math.round(v).toLocaleString());
        if (ids.detection_rate_recall) animateValue('kpi-ids', 0, ids.detection_rate_recall * 100, 800, v => v.toFixed(2) + '%');
        if (ids.dynamic_threshold_3sigma_mw) animateValue('kpi-threshold', 0, ids.dynamic_threshold_3sigma_mw, 800, v => Math.round(v).toLocaleString());
        if (fp.mape_percent)   animateValue('kpi-mape',       0, fp.mape_percent,        800, v => v.toFixed(2) + '%');
        if (ids.false_alarm_rate) animateValue('kpi-far',     0, ids.false_alarm_rate * 100, 800, v => v.toFixed(2) + '%');

        // Update ROC & F1 from metrics if available
        const rocEl = document.getElementById('kpi-roc');
        const f1El  = document.getElementById('kpi-f1');
        // We use fixed values since ROC needs full compute — they're already correct in HTML defaults
    } catch (err) {
        console.warn('Could not load live metrics from API:', err);
    }
}

// ─── 3. Animated counter helper ────────────────────────
function animateValue(elId, from, to, duration, formatter) {
    const el = document.getElementById(elId);
    if (!el) return;
    const start = performance.now();
    const tick = (now) => {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // cubic ease-out
        const current = from + (to - from) * eased;
        const formatted = formatter(current);
        // Preserve .unit child if present
        const unitEl = el.querySelector('.unit');
        if (unitEl) {
            el.childNodes[0].textContent = formatted;
        } else {
            el.textContent = formatted;
        }
        if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
}

// ─── 4. Attack Breakdown from API metrics ──────────────
async function loadAttackBreakdown() {
    try {
        const res = await fetch('/api/metrics');
        if (!res.ok) return;
        const data = await res.json();
        const breakdown = data.attack_type_breakdown || {};

        const mapping = {
            'FDI_SCALING':      { bar: document.querySelector('.fill-fdi-s'),  pct: null },
            'FDI_JITTER':       { bar: document.querySelector('.fill-fdi-j'),  pct: null },
            'DOS_LOAD_SURGE':   { bar: document.querySelector('.fill-dos'),    pct: null },
            'ENERGY_THEFT':     { bar: document.querySelector('.fill-theft'),  pct: null },
            'TELEMETRY_REPLAY': { bar: document.querySelector('.fill-replay'), pct: null }
        };

        // Get all .attack-bar-pct elements in order
        const pctEls = document.querySelectorAll('.attack-bar-pct');
        const keys = ['FDI_SCALING', 'FDI_JITTER', 'DOS_LOAD_SURGE', 'ENERGY_THEFT', 'TELEMETRY_REPLAY'];

        keys.forEach((key, i) => {
            const info = breakdown[key];
            const barEl = Object.values(mapping)[i].bar;
            const pctEl = pctEls[i];

            if (info && barEl && pctEl) {
                const pct = Math.round(info.recall * 100);
                setTimeout(() => {
                    barEl.style.width = pct + '%';
                    pctEl.textContent = pct + '%';
                }, 300 + i * 120);
            } else if (barEl && pctEl) {
                // Use data-pct fallback from HTML
                const fallbackPct = parseInt(barEl.getAttribute('data-pct') || '0');
                setTimeout(() => {
                    barEl.style.width = fallbackPct + '%';
                    pctEl.textContent = fallbackPct + '%';
                }, 300 + i * 120);
            }
        });
    } catch (err) {
        // Fallback to HTML data-pct values
        document.querySelectorAll('.attack-bar-fill').forEach((bar, i) => {
            const pct = parseInt(bar.getAttribute('data-pct') || '0');
            const pctEl = document.querySelectorAll('.attack-bar-pct')[i];
            setTimeout(() => {
                bar.style.width = pct + '%';
                if (pctEl) pctEl.textContent = pct + '%';
            }, 400 + i * 120);
        });
    }
}

// ─── 5. Telemetry Chart ────────────────────────────────
let telemetryChart = null;
let currentHoursLimit = 72;

async function initTelemetryChart() {
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentHoursLimit = parseInt(btn.getAttribute('data-hours'), 10);
            fetchAndRenderTelemetry();
        });
    });
    await fetchAndRenderTelemetry();
}

async function fetchAndRenderTelemetry() {
    try {
        const res = await fetch(`/api/historical?limit=${currentHoursLimit}`);
        const data = await res.json();
        if (!data.records || data.records.length === 0) return;

        const labels        = data.records.map(r => r.datetime);
        const reportedLoads = data.records.map(r => r.reported_mw);
        const forecastLoads = data.records.map(r => r.forecast_mw);
        const trueLoads     = data.records.map(r => r.true_mw);
        const alertPoints   = data.records.map(r => r.detected_attack === 1 ? r.reported_mw : null);

        const chartData = {
            labels,
            datasets: [
                {
                    label: 'Reported Telemetry (SCADA)',
                    data: reportedLoads,
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56,189,248,0.06)',
                    borderWidth: 2,
                    pointRadius: 0, pointHoverRadius: 4,
                    fill: true, tension: 0.2
                },
                {
                    label: 'ML Forecast Baseline (HistGradientBoosting)',
                    data: forecastLoads,
                    borderColor: '#00ff88',
                    borderWidth: 2,
                    borderDash: [5, 4],
                    pointRadius: 0, pointHoverRadius: 4, tension: 0.2
                },
                {
                    label: 'Ground Truth (Uncompromised)',
                    data: trueLoads,
                    borderColor: '#64748b',
                    borderWidth: 1.5,
                    pointRadius: 0, tension: 0.2
                },
                {
                    label: 'Cyber Intrusion Alert',
                    data: alertPoints,
                    showLine: false,
                    pointRadius: 5, pointHoverRadius: 8,
                    pointBackgroundColor: '#ff2e63',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 1.5,
                    pointStyle: 'crossRot'
                }
            ]
        };

        const chartOptions = {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(10,15,30,0.95)',
                    titleColor: '#f0f4fc',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1, padding: 12,
                    bodyFont: { family: 'JetBrains Mono', size: 12 },
                    callbacks: {
                        label: ctx => {
                            let label = ctx.dataset.label || '';
                            if (label) label += ': ';
                            if (ctx.parsed.y !== null)
                                label += Math.round(ctx.parsed.y).toLocaleString() + ' MW';
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#64748b', maxTicksLimit: 12, font: { family: 'JetBrains Mono', size: 10 } }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: {
                        color: '#64748b',
                        callback: val => (val / 1000).toFixed(0) + 'k MW',
                        font: { family: 'JetBrains Mono', size: 10 }
                    }
                }
            }
        };

        if (telemetryChart) {
            telemetryChart.data = chartData;
            telemetryChart.update('active');
        } else {
            const ctx = document.getElementById('telemetryChart').getContext('2d');
            telemetryChart = new Chart(ctx, { type: 'line', data: chartData, options: chartOptions });
        }
    } catch (err) {
        console.error('Failed to fetch telemetry:', err);
    }
}

// ─── 6. Residual Histogram (mini chart) ───────────────
let residualChart = null;

async function initResidualChart() {
    try {
        const res = await fetch('/api/historical?limit=500');
        const data = await res.json();
        if (!data.records || data.records.length === 0) return;

        const normalResiduals = data.records.filter(r => r.is_attack === 0).map(r => r.residual_mw);
        const attackResiduals = data.records.filter(r => r.is_attack === 1).map(r => r.residual_mw);

        const bins = 20;
        const maxVal = Math.max(...normalResiduals, ...attackResiduals, 500);
        const binSize = maxVal / bins;

        const buildHistogram = (arr) => {
            const counts = new Array(bins).fill(0);
            arr.forEach(v => {
                const b = Math.min(Math.floor(v / binSize), bins - 1);
                counts[b]++;
            });
            return counts;
        };

        const labels = Array.from({ length: bins }, (_, i) => Math.round(i * binSize));
        const normalCounts = buildHistogram(normalResiduals);
        const attackCounts = buildHistogram(attackResiduals);

        const ctx = document.getElementById('residualChart').getContext('2d');
        residualChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [
                    {
                        label: 'Normal Operation',
                        data: normalCounts,
                        backgroundColor: 'rgba(56,189,248,0.5)',
                        borderColor: '#38bdf8',
                        borderWidth: 1
                    },
                    {
                        label: 'Cyber Attack',
                        data: attackCounts,
                        backgroundColor: 'rgba(255,46,99,0.5)',
                        borderColor: '#ff2e63',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        labels: { color: '#94a3b8', font: { size: 10, family: 'JetBrains Mono' }, boxWidth: 12 }
                    }
                },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748b', font: { size: 9 } } },
                    y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748b', font: { size: 9 } } }
                }
            }
        });
    } catch (err) {
        console.warn('Residual chart skipped (no data yet):', err);
    }
}

// ─── 7. Attack Simulator ───────────────────────────────
function initAttackSimulator() {
    const baseSlider      = document.getElementById('baseline-load-slider');
    const baseDisplay     = document.getElementById('baseline-val-display');
    const intensitySlider = document.getElementById('attack-intensity-slider');
    const intensityDisplay = document.getElementById('intensity-val-display');
    const attackSelect    = document.getElementById('attack-type-select');
    const attackForm      = document.getElementById('attack-form');
    const btnLaunch       = document.getElementById('btn-launch-attack');

    baseSlider.addEventListener('input', e => {
        baseDisplay.textContent = Number(e.target.value).toLocaleString() + ' MW';
    });

    const updateIntensityLabel = () => {
        const val = intensitySlider.value;
        const type = attackSelect.value;
        if (type === 'ENERGY_THEFT')  intensityDisplay.textContent = `-${val}% (Bypass)`;
        else if (type === 'FDI_JITTER') intensityDisplay.textContent = `±${val}% Std`;
        else intensityDisplay.textContent = `+${val}%`;
    };

    intensitySlider.addEventListener('input', updateIntensityLabel);
    attackSelect.addEventListener('change', updateIntensityLabel);

    // Update timestamp on form open
    const updateTimestamp = () => {
        const now = new Date();
        document.getElementById('sim-timestamp').value = now.toISOString().replace('T', ' ').substring(0, 19);
    };
    attackSelect.addEventListener('change', updateTimestamp);

    attackForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        btnLaunch.disabled = true;
        btnLaunch.innerHTML = '<span class="spinner"></span> Analyzing Telemetry...';

        const attackType = attackSelect.value;
        const baselineMw = parseFloat(baseSlider.value);
        const intensity  = parseFloat(intensitySlider.value);
        const timestamp  = document.getElementById('sim-timestamp').value;

        const payload = { attack_type: attackType, baseline_mw: baselineMw, timestamp };
        if (attackType === 'FDI_SCALING')   payload.scale_factor = 1.0 + (intensity / 100.0);
        else if (attackType === 'FDI_JITTER')    payload.noise_std = intensity / 100.0;
        else if (attackType === 'DOS_LOAD_SURGE') payload.surge_pct = intensity / 100.0;
        else if (attackType === 'ENERGY_THEFT')   payload.theft_ratio = intensity / 100.0;
        else if (attackType === 'TELEMETRY_REPLAY') payload.freeze_val = baselineMw * 0.85;

        try {
            const res = await fetch('/api/simulate-attack', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            renderTriageResult(data);
        } catch (err) {
            console.error('Simulation request failed:', err);
        } finally {
            btnLaunch.disabled = false;
            btnLaunch.innerHTML = '<span class="btn-icon">⚡</span> Launch Cyber Attack Simulation';
        }
    });
}

function renderTriageResult(data) {
    const placeholder = document.querySelector('.triage-placeholder');
    const card        = document.getElementById('triage-card');
    const threatBadge = document.getElementById('threat-badge');

    if (placeholder) placeholder.style.display = 'none';
    card.classList.remove('hidden');
    // Re-trigger animation
    card.style.animation = 'none';
    card.offsetHeight; // reflow
    card.style.animation = '';

    const sim  = data.simulation_parameters;
    const resp = data.ml_defense_response;

    document.getElementById('metric-reported').textContent = sim.compromised_telemetry_mw.toLocaleString() + ' MW';
    document.getElementById('metric-pred').textContent     = resp.predicted_mw.toLocaleString() + ' MW';
    const deltaSign = sim.compromised_telemetry_mw >= resp.predicted_mw ? '+' : '-';
    document.getElementById('metric-residual').textContent = `${deltaSign}${resp.residual_mw.toLocaleString()} MW`;

    const ratio = (resp.residual_mw / resp.threshold_3sigma_mw).toFixed(1);
    const threshRatioEl = document.getElementById('thresh-ratio');
    const threshBarEl   = document.getElementById('thresh-bar');
    const threatBox     = document.getElementById('threat-class-box');

    // ── When ML Defense is turned OFF by the user ──
    if (!isMlDefenseActive) {
        threatBadge.className = 'badge-chip badge-alert';
        threatBadge.textContent = '❌ ML SHIELD OFF — ATTACK SUCCEEDED';
        threshRatioEl.className = 'text-crimson font-bold';
        threshRatioEl.textContent = 'DEFENSE BYPASSED (No ML Baseline Verification)';
        threshBarEl.style.width = '100%';
        threshBarEl.style.background = 'linear-gradient(90deg, #b91c1c, #dc2626)';
        threshBarEl.style.boxShadow = '0 0 16px rgba(220,38,38,0.8)';
        if (threatBox) threatBox.className = 'threat-classification-box';

        document.getElementById('classified-threat').textContent = sim.attack_type + " (UNCHECKED)";
        document.getElementById('classified-severity').textContent = 'CRITICAL RISK — SYSTEM BLIND';
        document.getElementById('mitigation-text').innerHTML = 
            '<strong class="text-crimson">ALERT: ML Defense is switched OFF!</strong> SCADA system accepted the falsified telemetry (' + 
            sim.compromised_telemetry_mw.toLocaleString() + ' MW) without validation. Grid dispatchers may trigger unnecessary peaker plants or trip sub-station breakers. <br><br>👉 <em>Turn ON the ML Defense Shield switch at the top to see how Machine Learning catches and stops this attack!</em>';

        showUnprotectedAttackModal(sim, resp);
        return;
    }

    if (resp.is_alarm) {
        threatBadge.className = 'badge-chip badge-alert';
        threatBadge.textContent = '⚠ INTRUSION DETECTED & INTERCEPTED';
        threshRatioEl.className = 'text-crimson font-bold';
        threshRatioEl.textContent = `${ratio}x EXCEEDED (limit: ${resp.threshold_3sigma_mw.toLocaleString()} MW)`;
        threshBarEl.style.width = Math.min(100, Math.max(5, ratio * 20)) + '%';
        threshBarEl.style.background = 'linear-gradient(90deg, #c41444, #ff2e63)';
        threshBarEl.style.boxShadow = '0 0 12px rgba(255,46,99,0.7)';
        if (threatBox) threatBox.className = 'threat-classification-box';
        document.getElementById('classified-threat').textContent = resp.threat_type;
        document.getElementById('classified-severity').textContent = resp.severity;
    } else {
        threatBadge.className = 'badge-chip badge-normal';
        threatBadge.textContent = '✓ NORMAL TELEMETRY';
        threshRatioEl.className = 'text-emerald font-bold';
        threshRatioEl.textContent = `Within 3σ Confidence Boundary`;
        threshBarEl.style.width = '18%';
        threshBarEl.style.background = 'linear-gradient(90deg, #00c96b, #00ff88)';
        threshBarEl.style.boxShadow = '0 0 10px rgba(0,255,136,0.5)';
        if (threatBox) threatBox.className = 'threat-classification-box status-ok';
        document.getElementById('classified-threat').textContent = 'BENIGN / NORMAL';
        document.getElementById('classified-severity').textContent = 'LOW';
    }

    document.getElementById('mitigation-text').textContent = resp.mitigation_action;

    // Pop up Alert Modal if an attack/alarm was triggered!
    if (resp.is_alarm) {
        showAttackAlertModal(sim, resp);
    }
}

function showAttackAlertModal(sim, resp) {
    const modal = document.getElementById('attack-alert-modal');
    if (!modal) return;

    document.getElementById('modal-attack-type').textContent = resp.threat_type;
    document.getElementById('modal-severity').textContent = resp.severity;
    document.getElementById('modal-reported-mw').textContent = sim.compromised_telemetry_mw.toLocaleString() + ' MW';
    document.getElementById('modal-predicted-mw').textContent = resp.predicted_mw.toLocaleString() + ' MW';
    
    const deltaSign = sim.compromised_telemetry_mw >= resp.predicted_mw ? '+' : '-';
    document.getElementById('modal-residual-mw').textContent = 
        `${deltaSign}${resp.residual_mw.toLocaleString()} MW (Threshold: ${resp.threshold_3sigma_mw.toLocaleString()} MW)`;

    // Custom explanation of what happened based on attack type
    let explainText = "";
    let stopActionText = "";

    if (resp.threat_type.includes("DOS") || resp.threat_type.includes("SURGE")) {
        explainText = "Attacker hijacked high-wattage IoT smart devices (EV chargers, smart HVACs) causing an artificial load spike of " + sim.compromised_telemetry_mw.toLocaleString() + " MW to trip circuit breakers.";
        stopActionText = "The trained HistGradientBoosting model anticipated a baseline of " + resp.predicted_mw.toLocaleString() + " MW. The 3-Sigma IDS caught the +" + resp.residual_mw.toLocaleString() + " MW anomaly, intercepted the fake surge telemetry, and prevented false generator dispatch.";
    } else if (resp.threat_type.includes("THEFT") || resp.threat_type.includes("BYPASS")) {
        explainText = "Attacker tampered with smart meter firmware to artificially depress reported demand to " + sim.compromised_telemetry_mw.toLocaleString() + " MW, concealing actual power consumption.";
        stopActionText = "The ML Forecaster recognized that daytime grid physics required at least " + resp.predicted_mw.toLocaleString() + " MW. The system flagged theft immediately and triggered automated audit logging.";
    } else if (resp.threat_type.includes("FDI") || resp.threat_type.includes("INJECTION")) {
        explainText = "Adversary infiltrated SCADA telemetry links and injected falsified power data to disrupt automatic generation control (AGC).";
        stopActionText = "The ML model compared the corrupted sensor values against historical temporal harmonics, detected the mathematical violation, and quarantined the compromised data channel.";
    } else {
        explainText = "Unusual telemetry distortion detected violating operational baseline boundaries.";
        stopActionText = "Isolation Forest and statistical 3-Sigma filters intercepted the abnormal reading and reverted the control system to safe ML predicted defaults.";
    }

    document.getElementById('modal-explain-text').textContent = explainText;
    document.getElementById('modal-stop-action').textContent = stopActionText;

    // Show modal with animation
    modal.classList.add('active');

    // Dismiss button
    const dismissBtn = document.getElementById('btn-modal-dismiss');
    if (dismissBtn) {
        dismissBtn.onclick = () => {
            modal.classList.remove('active');
        };
    }
}

function showUnprotectedAttackModal(sim, resp) {
    const modal = document.getElementById('attack-alert-modal');
    if (!modal) return;

    document.getElementById('modal-alert-title').textContent = '⚠️ CYBER ATTACK SUCCEEDED (ML SHIELD OFF)';
    document.getElementById('modal-attack-type').textContent = sim.attack_type + ' (UNCHECKED)';
    document.getElementById('modal-severity').textContent = 'CRITICAL — GRID COMPROMISED';
    document.getElementById('modal-reported-mw').textContent = sim.compromised_telemetry_mw.toLocaleString() + ' MW (ACCEPTED)';
    document.getElementById('modal-predicted-mw').textContent = 'BLIND / DISABLED (Expected: ~' + resp.predicted_mw.toLocaleString() + ' MW)';
    
    const deltaSign = sim.compromised_telemetry_mw >= resp.predicted_mw ? '+' : '-';
    document.getElementById('modal-residual-mw').textContent = 
        `DISCREPANCY: ${deltaSign}${resp.residual_mw.toLocaleString()} MW (IGNORED BY SCADA)`;

    document.getElementById('modal-explain-text').innerHTML = 
        `<span class="text-crimson"><strong>The Cyber Attack penetrated the Smart Grid!</strong></span> Because the ML Defense Shield was toggled <strong>OFF</strong>, the legacy SCADA system trusted the tampered sensor telemetry (${sim.compromised_telemetry_mw.toLocaleString()} MW) without mathematical anomaly verification.`;

    document.getElementById('modal-stop-action').innerHTML = 
        `<strong>Consequence:</strong> Automated Generation Control (AGC) dispatched incorrect generation reserves, creating physical frequency instability, voltage collapse, or potential regional blackout.<br><br>👉 <strong>Toggle the ML Defense Shield to "ACTIVE" at the top</strong> to see how the HistGradientBoosting + 3σ model instantly detects and halts this exact attack!`;

    modal.classList.add('active');

    const dismissBtn = document.getElementById('btn-modal-dismiss');
    if (dismissBtn) {
        dismissBtn.textContent = 'Acknowledge Demonstration';
        dismissBtn.onclick = () => {
            modal.classList.remove('active');
            // Reset button text
            setTimeout(() => {
                dismissBtn.textContent = 'Acknowledge & Resume Monitoring';
                document.getElementById('modal-alert-title').textContent = 'CYBER ATTACK INTERCEPTED!';
            }, 300);
        };
    }
}

// ─── 8. Diagnostics Gallery ────────────────────────────
function initDiagnosticsGallery() {
    const tabs    = document.querySelectorAll('.tab-btn');
    const imgEl   = document.getElementById('active-diagnostic-img');
    const captEl  = document.getElementById('active-diagnostic-caption');

    const captions = {
        'forecast_vs_actual.png': '<strong>Figure 1: Actual vs. ML Forecasted Consumption</strong> — 14-day zoomed evaluation horizon comparing actual power demand with HistGradientBoosting Regressor predictions and absolute residual error bounds against the dynamic 3σ alert threshold.',
        'residual_analysis.png':  '<strong>Figure 2: Residual Error Analysis &amp; Dynamic 3σ Boundary</strong> — Probability density of forecast error under normal grid operation vs. cyber-attack episodes, demonstrating clean 99.9% statistical boundary separation.',
        'cyber_attack_detection.png': '<strong>Figure 3: Full-Horizon Cyber Attack Detection</strong> — Real-time telemetry feed featuring labeled FDI scaling, high-frequency jitter, DoS load surge, energy theft, and replay attack episodes flagged by the ML defense system.',
        'feature_importance.png': '<strong>Figure 4: Permutation Feature Importance</strong> — Quantifying the impact of calendar harmonics, cyclic sin/cos encoders, autoregressive lags (lag_1, lag_24, lag_168), and rolling 24h baseline statistics on forecast accuracy.',
        'roc_confusion_matrix.png': '<strong>Figure 5: Intrusion Detection Confusion Matrix &amp; ROC Curve</strong> — Demonstrates 94.95% attack detection rate with only 4.09% false alarm rate on holdout smart grid evaluation telemetry. ROC-AUC = 0.9793.'
    };

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            const filename = tab.getAttribute('data-img');
            imgEl.style.opacity = '0';
            setTimeout(() => {
                imgEl.src = `/reports/figures/${filename}`;
                imgEl.onload = () => { imgEl.style.opacity = '1'; };
                captEl.innerHTML = captions[filename] || '';
            }, 150);
        });
    });
}

// ─── 9. Incident Log Table ─────────────────────────────
let allIncidentRecords = [];

async function loadIncidentLog() {
    try {
        const res = await fetch('/api/historical?limit=20&offset=380');
        const data = await res.json();
        allIncidentRecords = data.records || [];
        renderIncidentTable(allIncidentRecords);
    } catch (err) {
        console.error('Failed to load incident table:', err);
    }
}

function renderIncidentTable(records) {
    const tbody = document.getElementById('incident-tbody');

    if (!records || records.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-center">No security incidents found.</td></tr>';
        return;
    }

    tbody.innerHTML = records.map(r => {
        const isAlert    = r.detected_attack === 1;
        const isRealAtk  = r.is_attack === 1;
        const statusBadge = isAlert
            ? '<span class="badge-alert">⚠ ALARM</span>'
            : '<span class="badge-normal">✓ BENIGN</span>';
        const severity = isAlert
            ? '<strong class="text-crimson">CRITICAL</strong>'
            : '<span class="text-secondary">LOW</span>';
        const residualClass = isAlert ? 'text-crimson font-bold' : '';

        return `
            <tr>
                <td>${r.datetime}</td>
                <td><strong>${r.reported_mw.toLocaleString()} MW</strong></td>
                <td>${r.forecast_mw.toLocaleString()} MW</td>
                <td class="${residualClass}">${r.residual_mw.toLocaleString()} MW</td>
                <td><span class="font-mono">${r.attack_type}</span></td>
                <td>${isRealAtk ? '<span class="text-crimson font-bold">ATTACK</span>' : '<span class="text-secondary">BENIGN</span>'}</td>
                <td>${statusBadge}</td>
                <td>${severity}</td>
            </tr>
        `;
    }).join('');
}

// ─── 10. Table Search ──────────────────────────────────
function initTableSearch() {
    const searchEl = document.getElementById('log-search');
    if (!searchEl) return;
    searchEl.addEventListener('input', e => {
        const q = e.target.value.toLowerCase();
        const filtered = allIncidentRecords.filter(r =>
            r.datetime.toLowerCase().includes(q) ||
            r.attack_type.toLowerCase().includes(q) ||
            String(r.reported_mw).includes(q) ||
            (r.detected_attack === 1 && 'alarm'.includes(q)) ||
            (r.detected_attack === 0 && 'benign'.includes(q))
        );
        renderIncidentTable(filtered);
    });
}

// ─── 11. Theme Switcher (White / Dark) ────────────────
function initThemeToggle() {
    const btn = document.getElementById('btn-theme-toggle');
    if (!btn) return;
    const themeIcon = btn.querySelector('.theme-icon');
    const themeLabel = btn.querySelector('.theme-label');

    // Check saved theme or default to light-theme
    const savedTheme = localStorage.getItem('gridguard_theme') || 'light';
    if (savedTheme === 'dark') {
        document.body.classList.remove('light-theme');
        document.body.classList.add('dark-theme');
        themeIcon.textContent = '☀️';
        themeLabel.textContent = 'Light';
    } else {
        document.body.classList.remove('dark-theme');
        document.body.classList.add('light-theme');
        themeIcon.textContent = '🌙';
        themeLabel.textContent = 'Dark';
    }

    btn.addEventListener('click', () => {
        const isLight = document.body.classList.contains('light-theme');
        if (isLight) {
            document.body.classList.remove('light-theme');
            document.body.classList.add('dark-theme');
            themeIcon.textContent = '☀️';
            themeLabel.textContent = 'Light';
            localStorage.setItem('gridguard_theme', 'dark');
        } else {
            document.body.classList.remove('dark-theme');
            document.body.classList.add('light-theme');
            themeIcon.textContent = '🌙';
            themeLabel.textContent = 'Dark';
            localStorage.setItem('gridguard_theme', 'light');
        }
    });
}

// ─── 12. ML Defense Shield ON / OFF Toggle ────────────
function initDefenseToggle() {
    const toggleInput = document.getElementById('ml-defense-toggle');
    const statusText  = document.getElementById('ml-status-text');
    const wrapper     = document.querySelector('.ml-toggle-wrapper');
    if (!toggleInput) return;

    toggleInput.addEventListener('change', () => {
        isMlDefenseActive = toggleInput.checked;
        if (isMlDefenseActive) {
            statusText.textContent = 'ACTIVE (SHIELD ON)';
            statusText.className = 'status-indicator active';
            if (wrapper) wrapper.classList.remove('disabled-mode');
        } else {
            statusText.textContent = 'DISABLED (UNPROTECTED)';
            statusText.className = 'status-indicator disabled';
            if (wrapper) wrapper.classList.add('disabled-mode');
        }
    });
}
