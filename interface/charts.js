// ============================================================
// ElectroMacroDiff V5.2 Dashboard — Charts Module
// Chart.js visualizations for docking, ADMET, and pipeline data
// ============================================================

const chartDefaults = { color: '#9fa4c4', borderColor: 'rgba(100,120,255,0.13)' };
Chart.defaults.color = chartDefaults.color;
Chart.defaults.borderColor = chartDefaults.borderColor;
Chart.defaults.font.family = "'Inter', sans-serif";
const charts = {};

function makeHist(data, bins = 25) {
  const valid = data.filter(v => v != null && !isNaN(v));
  if (!valid.length) return { labels: [], counts: [] };
  const min = Math.min(...valid), max = Math.max(...valid);
  const step = (max - min) / bins || 1;
  const labels = [], counts = new Array(bins).fill(0);
  for (let i = 0; i < bins; i++) labels.push((min + step * (i + 0.5)).toFixed(2));
  valid.forEach(v => { const b = Math.min(Math.floor((v - min) / step), bins - 1); counts[b]++; });
  return { labels, counts };
}

// ---- Docking Charts ----
let dockingChartsInit = false;
function initDockingCharts() {
  if (dockingChartsInit) return; dockingChartsInit = true;
  const scored = ALL.filter(c => c.has_docking && c.best_score != null);

  // Histogram
  const hist = makeHist(scored.map(c => c.best_score), 30);
  charts.dockHist = new Chart(document.getElementById('dockingHistChart'), {
    type: 'bar', data: { labels: hist.labels, datasets: [{ label: 'Count', data: hist.counts,
      backgroundColor: 'rgba(99,102,241,0.55)', borderColor: 'rgba(129,140,248,0.7)', borderWidth: 1, borderRadius: 3 }] },
    options: { responsive: true, plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: 'Vina Score (kcal/mol)' } }, y: { title: { display: true, text: 'Count' } } } }
  });

  // Scatter: Pocket Fit vs Vina
  const pocketScored = ALL.filter(c => c.has_pocket_score && c.best_score != null);
  charts.scatterDock = new Chart(document.getElementById('scatterDockChart'), {
    type: 'scatter', data: { datasets: [{
      label: 'Candidates', data: pocketScored.map(c => ({ x: c.best_score, y: c.pocket_electronic_fit_score })),
      backgroundColor: pocketScored.map(c => c.selection_tier === 'final_candidate' ? 'rgba(52,211,153,0.65)' : 'rgba(99,102,241,0.45)'),
      pointRadius: 3.5, pointHoverRadius: 6 }] },
    options: { responsive: true, plugins: { legend: { display: false },
      tooltip: { callbacks: { label: (ctx) => { const c = pocketScored[ctx.dataIndex]; return `${c.candidate_id}: Vina=${c.best_score}, Fit=${c.pocket_electronic_fit_score?.toFixed(3)}`; } } } },
      scales: { x: { title: { display: true, text: 'Vina Score (kcal/mol)' }, reverse: true }, y: { title: { display: true, text: 'Pocket Electronic Fit Score' } } } }
  });

  // Radar: Pocket components top 10
  const top10 = ALL.slice(0, 10);
  charts.radarPocket = new Chart(document.getElementById('radarPocketChart'), {
    type: 'radar', data: {
      labels: ['Electrostatic', 'H-Bond', 'Hydrophobic', 'Aromatic', 'Pocket Fit'],
      datasets: top10.slice(0, 5).map((c, i) => ({
        label: c.candidate_id.slice(-6), fill: true, backgroundColor: [`rgba(99,102,241,0.1)`,`rgba(34,211,238,0.1)`,`rgba(52,211,153,0.1)`,`rgba(251,191,36,0.1)`,`rgba(167,139,250,0.1)`][i],
        borderColor: ['#6366f1','#22d3ee','#34d399','#fbbf24','#a78bfa'][i], pointBackgroundColor: ['#6366f1','#22d3ee','#34d399','#fbbf24','#a78bfa'][i],
        data: [c.electrostatic_score, c.hbond_score, c.hydrophobic_score, c.aromatic_score, c.pocket_electronic_fit_score] })) },
    options: { responsive: true, scales: { r: { min: 0, max: 1, ticks: { stepSize: 0.2 } } },
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } } } }
  });

  // Ring size vs score
  const ringSizes = {};
  scored.forEach(c => { const rs = c.max_ring_size; if (rs) { if (!ringSizes[rs]) ringSizes[rs] = []; ringSizes[rs].push(c.best_score); } });
  const rsLabels = Object.keys(ringSizes).sort((a, b) => a - b);
  charts.ringSize = new Chart(document.getElementById('ringSizeChart'), {
    type: 'bar', data: { labels: rsLabels.map(r => `${r}-mem`),
      datasets: [{ label: 'Avg Vina Score', data: rsLabels.map(r => (ringSizes[r].reduce((a, b) => a + b, 0) / ringSizes[r].length).toFixed(2)),
        backgroundColor: 'rgba(34,211,238,0.5)', borderColor: 'rgba(34,211,238,0.7)', borderWidth: 1, borderRadius: 3 },
      { label: 'Count', data: rsLabels.map(r => ringSizes[r].length), backgroundColor: 'rgba(99,102,241,0.3)', borderRadius: 3, yAxisID: 'y1' }] },
    options: { responsive: true, scales: { y: { title: { display: true, text: 'Avg Vina Score' } }, y1: { position: 'right', title: { display: true, text: 'Count' }, grid: { drawOnChartArea: false } } },
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 10 } } } }
  });
}

// ---- ADMET Charts ----
let admetChartsInit = false;
function initAdmetCharts() {
  if (admetChartsInit) return; admetChartsInit = true;
  const barOpts = (xlabel) => ({ responsive: true, plugins: { legend: { display: false } },
    scales: { x: { title: { display: true, text: xlabel } }, y: { title: { display: true, text: 'Count' } } } });

  // QED
  const qh = makeHist(ALL.map(c => c.qed), 25);
  charts.qedHist = new Chart(document.getElementById('qedHistChart'), {
    type: 'bar', data: { labels: qh.labels, datasets: [{ data: qh.counts, backgroundColor: 'rgba(52,211,153,0.5)', borderRadius: 3 }] }, options: barOpts('QED')
  });

  // MW
  const mh = makeHist(ALL.map(c => c.mw), 25);
  charts.mwHist = new Chart(document.getElementById('mwHistChart'), {
    type: 'bar', data: { labels: mh.labels, datasets: [{ data: mh.counts, backgroundColor: 'rgba(99,102,241,0.5)', borderRadius: 3 }] }, options: barOpts('Molecular Weight (Da)')
  });

  // LogP
  const lh = makeHist(ALL.map(c => c.logp), 25);
  charts.logpHist = new Chart(document.getElementById('logpHistChart'), {
    type: 'bar', data: { labels: lh.labels, datasets: [{ data: lh.counts, backgroundColor: 'rgba(251,191,36,0.5)', borderRadius: 3 }] }, options: barOpts('LogP')
  });

  // SA
  const sh = makeHist(ALL.map(c => c.sa_score), 25);
  charts.saHist = new Chart(document.getElementById('saHistChart'), {
    type: 'bar', data: { labels: sh.labels, datasets: [{ data: sh.counts, backgroundColor: 'rgba(167,139,250,0.5)', borderRadius: 3 }] }, options: barOpts('SA Score')
  });

  // ADMET vs QED scatter
  charts.admetQed = new Chart(document.getElementById('admetQedScatter'), {
    type: 'scatter', data: { datasets: [
      { label: '0 violations', data: ALL.filter(c => c.lipinski_violations === 0).map(c => ({ x: c.qed, y: c.admet_score })),
        backgroundColor: 'rgba(52,211,153,0.5)', pointRadius: 2.5 },
      { label: '1+ violations', data: ALL.filter(c => c.lipinski_violations > 0).map(c => ({ x: c.qed, y: c.admet_score })),
        backgroundColor: 'rgba(251,113,133,0.6)', pointRadius: 3 }
    ] },
    options: { responsive: true, scales: { x: { title: { display: true, text: 'QED' } }, y: { title: { display: true, text: 'ADMET Score' } } },
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 10 } } } }
  });

  // ADMET Summary Grid
  buildAdmetSummary();
}

function buildAdmetSummary() {
  const el = document.getElementById('admetSummary'); if (!el) return;
  const props = [
    { name: 'Molecular Weight', key: 'mw', min: 100, max: 700, unit: 'Da' },
    { name: 'LogP', key: 'logp', min: -2, max: 8, unit: '' },
    { name: 'QED', key: 'qed', min: 0, max: 1, unit: '' },
    { name: 'SA Score', key: 'sa_score', min: 1, max: 7, unit: '' },
    { name: 'ADMET Score', key: 'admet_score', min: 0, max: 1, unit: '' },
    { name: 'TPSA', key: 'tpsa', min: 0, max: 200, unit: 'Å²' },
    { name: 'H-Bond Donors', key: 'hbd', min: 0, max: 6, unit: '' },
    { name: 'H-Bond Acceptors', key: 'hba', min: 0, max: 12, unit: '' },
    { name: 'Synthesis Score', key: 'synthesis_score', min: 0, max: 1, unit: '' },
    { name: 'Pocket Fit Score', key: 'pocket_electronic_fit_score', min: 0, max: 1, unit: '' },
  ];
  el.innerHTML = props.map(p => {
    const vals = ALL.map(c => c[p.key]).filter(v => v != null && !isNaN(v));
    if (!vals.length) return '';
    const mn = Math.min(...vals), mx = Math.max(...vals), avg = vals.reduce((a, b) => a + b, 0) / vals.length;
    const pct = ((avg - p.min) / (p.max - p.min) * 100).toFixed(0);
    return `<div class="admet-stat-row"><div class="admet-stat-name">${p.name}</div>
      <div class="admet-bar-track"><div class="admet-bar-fill" style="width:${Math.min(100, Math.max(5, pct))}%"></div></div>
      <div class="admet-stat-vals"><span>Min: ${mn.toFixed(2)}</span><span>Avg: ${avg.toFixed(2)}</span><span>Max: ${mx.toFixed(2)}</span></div></div>`;
  }).join('');
}

// ---- Pipeline Charts ----
let pipelineChartsInit = false;
function initPipelineCharts() {
  if (pipelineChartsInit) return; pipelineChartsInit = true;

  // Generator pie
  charts.genPie = new Chart(document.getElementById('genPieChart'), {
    type: 'doughnut', data: { labels: ['Macrocycle Linker', 'RDKit', 'SELFIES'],
      datasets: [{ data: [2935, 1249, 2590], backgroundColor: ['rgba(99,102,241,0.7)','rgba(34,211,238,0.7)','rgba(52,211,153,0.7)'],
        borderColor: ['#6366f1','#22d3ee','#34d399'], borderWidth: 2 }] },
    options: { responsive: true, plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, padding: 16 } } } }
  });

  // Scoring radar for top 10
  const top10 = ALL.slice(0, 10);
  charts.scoringRadar = new Chart(document.getElementById('scoringRadar'), {
    type: 'radar', data: {
      labels: ['Docking', 'ADMET', 'Synthesis', 'Safety', 'Novelty', 'Pocket Fit'],
      datasets: top10.slice(0, 4).map((c, i) => {
        const cols = ['#6366f1','#22d3ee','#34d399','#fbbf24'];
        // Normalize docking to 0-1 range
        const normDock = Math.min(1, Math.abs(c.best_score || 0) / 13);
        return { label: c.candidate_id.slice(-6), fill: true,
          backgroundColor: cols[i] + '18', borderColor: cols[i], pointBackgroundColor: cols[i],
          data: [normDock, c.admet_score, c.synthesis_score, c.safety_proxy_score, c.novelty_score, c.pocket_electronic_fit_score] };
      }) },
    options: { responsive: true, scales: { r: { min: 0, max: 1, ticks: { stepSize: 0.2 } } },
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } } } }
  });

  // Ring size distribution bar
  const ringCounts = {};
  ALL.forEach(c => { const rs = c.max_ring_size; if (rs) ringCounts[rs] = (ringCounts[rs] || 0) + 1; });
  const rsLabels = Object.keys(ringCounts).sort((a, b) => a - b);
  charts.ringSizeBar = new Chart(document.getElementById('ringSizeBarChart'), {
    type: 'bar', data: { labels: rsLabels.map(r => `${r}-membered`),
      datasets: [{ label: 'Count', data: rsLabels.map(r => ringCounts[r]),
        backgroundColor: rsLabels.map((_, i) => `hsla(${220 + i * 12}, 70%, 60%, 0.6)`), borderRadius: 4 }] },
    options: { responsive: true, plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: 'Ring Size' } }, y: { title: { display: true, text: 'Count' } } } }
  });
}
