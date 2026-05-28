// ElectroMacroDiff dashboard application.
// This file displays only values exported in data.js from project outputs.
// Missing values are rendered as "not available"; no benchmark/result values
// are invented in the browser.

const S = typeof REAL_PROJECT_SUMMARY !== "undefined" ? REAL_PROJECT_SUMMARY : {};
const ALL = typeof REAL_ALL_CANDIDATES !== "undefined" && Array.isArray(REAL_ALL_CANDIDATES)
  ? REAL_ALL_CANDIDATES
  : [];
const TOP = typeof REAL_TOP_CANDIDATES !== "undefined" && Array.isArray(REAL_TOP_CANDIDATES)
  ? REAL_TOP_CANDIDATES
  : ALL.slice(0, 20);
const BRANCHES = typeof REAL_BRANCH_METRICS !== "undefined" && Array.isArray(REAL_BRANCH_METRICS)
  ? REAL_BRANCH_METRICS
  : [];
const ATTEMPTS = typeof REAL_ATTEMPT_STATUS_COUNTS !== "undefined" ? REAL_ATTEMPT_STATUS_COUNTS : {};
const DOCKING_HIST = typeof REAL_DOCKING_HIST !== "undefined" && Array.isArray(REAL_DOCKING_HIST)
  ? REAL_DOCKING_HIST
  : [];
const ADMET = typeof REAL_ADMET_SUMMARY !== "undefined" ? REAL_ADMET_SUMMARY : {};
const GRID = typeof REAL_GRID !== "undefined" ? REAL_GRID : {};
const POCKET = typeof REAL_POCKET_PROFILE !== "undefined" ? REAL_POCKET_PROFILE : {};
const POSES = typeof REAL_POSE_DATA !== "undefined" ? REAL_POSE_DATA : {};
const POSE_PATHS = typeof REAL_POSE_PATHS !== "undefined" ? REAL_POSE_PATHS : {};
const PROTEIN_PDB = typeof REAL_PROTEIN_PDB !== "undefined" ? REAL_PROTEIN_PDB : "";

const PAGE_SIZE = 50;
const CHARTS = {};
let filteredData = [...ALL];
let currentPage = 1;
let viewer3d = null;
let selectedCandidateIndex = 0;
let selectedCandidate = ALL[0] || null;
const VIEWER_STATE = {
  proteinModel: null,
  ligandModel: null,
  poseCache: { ...POSES },
  activePoseText: null,
  poseSource: "not_loaded",
};

function byId(id) {
  return document.getElementById(id);
}

function hasValue(value) {
  return value !== null && value !== undefined && value !== "" && Number.isFinite(value) !== false;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function numberValue(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function fmt(value, digits = 3) {
  const n = numberValue(value);
  if (n === null) return "not available";
  return n.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function fmtLoose(value, digits = 3) {
  const n = numberValue(value);
  if (n === null) return "not available";
  return n.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function fmtInt(value) {
  const n = numberValue(value);
  return n === null ? "not available" : Math.round(n).toLocaleString();
}

function fmtPct(value, digits = 2) {
  const n = numberValue(value);
  return n === null ? "not available" : `${fmt(n, digits)}%`;
}

function fmtScore(value) {
  const n = numberValue(value);
  return n === null ? "not available" : n.toFixed(1);
}

function setText(id, value) {
  const el = byId(id);
  if (el) el.textContent = value;
}

function setHtml(id, html) {
  const el = byId(id);
  if (el) el.innerHTML = html;
}

function truthNote() {
  return S.truth_note || "All displayed values are loaded from project output files.";
}

function candidateRank(candidate, fallbackIndex = 0) {
  return candidate?.rank ?? fallbackIndex + 1;
}

function candidateLabel(candidate) {
  return candidate?.candidate_id || "not available";
}

function candidateScore(candidate, key, digits = 3) {
  return fmt(candidate?.[key], digits);
}

function sourceFileName(pathText) {
  if (!pathText) return "project output file";
  const pieces = String(pathText).split(/[\\/]/);
  return pieces[pieces.length - 1] || "project output file";
}

function showEmpty(container, message) {
  if (container) {
    container.innerHTML = `<div class="viewer-overlay-text" style="position:relative;min-height:96px">${escapeHtml(message)}</div>`;
  }
}

function makeStat(label, value) {
  return `
    <div class="modal-stat">
      <div class="modal-stat-label">${escapeHtml(label)}</div>
      <div class="modal-stat-value">${escapeHtml(value ?? "not available")}</div>
    </div>`;
}

function makePocketStat(label, value) {
  return `
    <div class="pocket-stat">
      <div class="stat-label">${escapeHtml(label)}</div>
      <div class="stat-value">${escapeHtml(value ?? "not available")}</div>
    </div>`;
}

function tierText(value) {
  if (!value) return "not available";
  return String(value).replace(/_/g, " ");
}

function tierBadge(value) {
  const tier = value || "";
  const cls = tier === "final_candidate" || tier === "primary_candidate" ? "tier-final" : "tier-backup";
  return `<span class="tier-badge ${cls}">${escapeHtml(tierText(tier))}</span>`;
}

function availablePoseCount() {
  const cacheCount = Object.keys(VIEWER_STATE.poseCache || {}).length;
  const pathCount = Object.keys(POSE_PATHS || {}).length;
  return Math.max(cacheCount, pathCount);
}

// Navigation
function showSection(id, el) {
  document.querySelectorAll(".section").forEach((section) => section.classList.remove("active"));
  document.querySelectorAll(".nav-link").forEach((link) => link.classList.remove("active"));

  const target = byId(id);
  if (target) target.classList.add("active");
  if (el) {
    el.classList.add("active");
  } else {
    document.querySelector(`.nav-link[href="#${id}"]`)?.classList.add("active");
  }

  if (id === "viewer3d") init3DViewer();
  if (id === "docking") initDockingSection();
  if (id === "admet") initAdmetSection();
  if (id === "pipeline") initPipelineSection();

  byId("mobileNav")?.classList.remove("open");
}

function toggleMobileMenu() {
  byId("mobileNav")?.classList.toggle("open");
}

// Overview
function populateHeaderAndKpis() {
  setText("kpiLigands", fmtInt(S.curated_jak2_ligands));
  setText("kpiCandidates", fmtInt(S.ranked_candidates ?? ALL.length));
  setText("kpiBestScore", fmtScore(S.best_vina_gpu_score));
  setText("kpiBestFit", fmtLoose(S.best_pocket_electronic_fit_score, 3));
  setText("kpiTopScore", fmtLoose(S.top_ranked_final_score, 3));

  const testsPassed = S.validation?.unit_tests_passed;
  const testsTotal = S.validation?.unit_tests_total;
  const validationPassed = S.validation?.passed;
  const validationTotal = S.validation?.total;
  const unitText = testsPassed != null && testsTotal != null
    ? `${testsPassed}/${testsTotal}`
    : "not available";
  const valText = validationPassed != null && validationTotal != null
    ? `; validation ${validationPassed}/${validationTotal}`
    : "";
  setText("kpiTests", unitText);
  const testsLabel = byId("kpiTests")?.parentElement?.querySelector(".kpi-label");
  if (testsLabel) testsLabel.textContent = `Unit Tests Passed${valText}`;

  const tableCount = Math.min(20, ALL.length);
  setText("tableCount", `Showing top ${tableCount.toLocaleString()} of ${ALL.length.toLocaleString()}`);
  setText("filterStats", `Showing ${ALL.length.toLocaleString()} project candidates`);
  setText("candidateSectionBadge", `${ALL.length.toLocaleString()} candidates`);
  setText("topSourceBadge", `From ${sourceFileName(S.ranking_source)}`);
  setText("admetBadge", `${ALL.length.toLocaleString()} ranked candidates`);
}

function buildHeroCandidates() {
  const el = byId("heroCandidates");
  if (!el) return;
  const rows = TOP.slice(0, 5);
  if (!rows.length) {
    showEmpty(el, "No ranked candidate rows were exported.");
    return;
  }

  el.innerHTML = rows.map((candidate, localIndex) => {
    const globalIndex = ALL.findIndex((row) => row.candidate_id === candidate.candidate_id);
    const idx = globalIndex >= 0 ? globalIndex : localIndex;
    return `
      <div class="hero-cand" onclick="openCandModal(${idx})">
        <div class="hero-cand-rank">Rank #${escapeHtml(candidateRank(candidate, idx))}</div>
        <div class="hero-cand-id">${escapeHtml(candidateLabel(candidate))}</div>
        <div class="hero-cand-score">${escapeHtml(candidateScore(candidate, "pocket_guided_final_score", 4))}</div>
        <div class="hero-cand-label">Pocket-guided final score</div>
        <div class="hero-cand-grid">
          <div class="hero-cand-stat"><div class="hero-cand-stat-label">Vina</div><div class="hero-cand-stat-val">${escapeHtml(fmtScore(candidate.best_score))}</div></div>
          <div class="hero-cand-stat"><div class="hero-cand-stat-label">Pocket Fit</div><div class="hero-cand-stat-val">${escapeHtml(fmtLoose(candidate.pocket_electronic_fit_score, 3))}</div></div>
          <div class="hero-cand-stat"><div class="hero-cand-stat-label">ADMET</div><div class="hero-cand-stat-val">${escapeHtml(fmtLoose(candidate.admet_score, 3))}</div></div>
          <div class="hero-cand-stat"><div class="hero-cand-stat-label">QED</div><div class="hero-cand-stat-val">${escapeHtml(fmtLoose(candidate.qed, 3))}</div></div>
          <div class="hero-cand-stat"><div class="hero-cand-stat-label">MW</div><div class="hero-cand-stat-val">${escapeHtml(fmtLoose(candidate.mw, 1))}</div></div>
          <div class="hero-cand-stat"><div class="hero-cand-stat-label">Ring</div><div class="hero-cand-stat-val">${escapeHtml(fmtInt(candidate.max_ring_size))}</div></div>
        </div>
        <button class="hero-cand-btn" type="button">View real row details</button>
      </div>`;
  }).join("");
}

function buildOverviewTable() {
  const tb = byId("overviewTableBody");
  if (!tb) return;
  const rows = TOP.slice(0, 20);
  if (!rows.length) {
    tb.innerHTML = `<tr><td colspan="9">No ranked candidate rows were exported.</td></tr>`;
    return;
  }

  tb.innerHTML = rows.map((candidate, localIndex) => {
    const idx = ALL.findIndex((row) => row.candidate_id === candidate.candidate_id);
    const globalIndex = idx >= 0 ? idx : localIndex;
    return `
      <tr onclick="openCandModal(${globalIndex})">
        <td>${escapeHtml(candidateRank(candidate, globalIndex))}</td>
        <td>${escapeHtml(candidateLabel(candidate))}</td>
        <td>${escapeHtml(fmtScore(candidate.best_score))}</td>
        <td>${escapeHtml(fmtLoose(candidate.pocket_electronic_fit_score, 3))}</td>
        <td>${escapeHtml(fmtLoose(candidate.pocket_guided_final_score, 4))}</td>
        <td>${escapeHtml(fmtLoose(candidate.mw, 1))}</td>
        <td>${escapeHtml(fmtLoose(candidate.qed, 3))}</td>
        <td>${tierBadge(candidate.selection_tier)}</td>
        <td><button class="btn-view" type="button" onclick="event.stopPropagation();goTo3D(${globalIndex})">3D</button></td>
      </tr>`;
  }).join("");
}

// Candidate filtering and table
function getFilterValues() {
  return {
    sortBy: byId("sortBy")?.value || "pocket_guided_final_score",
    tier: byId("tierFilter")?.value || "",
    decision: byId("decisionFilter")?.value || "",
    lipinski: byId("lipinskiFilter")?.value || "",
    macrocycle: byId("macrocycleFilter")?.value || "",
    minVina: numberValue(byId("minVina")?.value),
    maxVina: numberValue(byId("maxVina")?.value),
    minQed: numberValue(byId("minQed")?.value),
    minPocketFit: numberValue(byId("minPocketFit")?.value),
    search: String(byId("searchText")?.value || "").trim().toLowerCase(),
  };
}

function candidateMatchesFilters(candidate, filters) {
  if (filters.tier && candidate.selection_tier !== filters.tier) return false;
  if (filters.decision && candidate.decision !== filters.decision) return false;
  if (filters.lipinski !== "") {
    const violations = numberValue(candidate.lipinski_violations);
    if (violations === null || violations > Number(filters.lipinski)) return false;
  }
  if (filters.macrocycle === "true" && candidate.has_macrocycle !== true) return false;
  if (filters.macrocycle === "false" && candidate.has_macrocycle !== false) return false;

  const vina = numberValue(candidate.best_score);
  if (filters.minVina !== null && (vina === null || vina < filters.minVina)) return false;
  if (filters.maxVina !== null && (vina === null || vina > filters.maxVina)) return false;

  const qed = numberValue(candidate.qed);
  if (filters.minQed !== null && (qed === null || qed < filters.minQed)) return false;

  const pocketFit = numberValue(candidate.pocket_electronic_fit_score);
  if (filters.minPocketFit !== null && (pocketFit === null || pocketFit < filters.minPocketFit)) return false;

  if (filters.search) {
    const haystack = [
      candidate.candidate_id,
      candidate.smiles,
      candidate.inchikey,
      candidate.parent_mol_id,
      candidate.source,
    ].filter(Boolean).join(" ").toLowerCase();
    if (!haystack.includes(filters.search)) return false;
  }
  return true;
}

function sortCandidates(rows, sortBy) {
  const ascendingKeys = new Set(["best_score", "mw", "sa_score"]);
  const ascending = ascendingKeys.has(sortBy);
  rows.sort((a, b) => {
    const av = numberValue(a[sortBy]);
    const bv = numberValue(b[sortBy]);
    if (av === null && bv === null) return candidateRank(a) - candidateRank(b);
    if (av === null) return 1;
    if (bv === null) return -1;
    return ascending ? av - bv : bv - av;
  });
}

function applyFilters() {
  const filters = getFilterValues();
  filteredData = ALL.filter((candidate) => candidateMatchesFilters(candidate, filters));
  sortCandidates(filteredData, filters.sortBy);
  currentPage = 1;
  renderCandTable();
  setText("filterStats", `Showing ${filteredData.length.toLocaleString()} of ${ALL.length.toLocaleString()} project candidates`);
}

function resetFilters() {
  ["sortBy", "tierFilter", "decisionFilter", "lipinskiFilter", "macrocycleFilter"].forEach((id) => {
    const el = byId(id);
    if (el) el.selectedIndex = 0;
  });
  ["minVina", "maxVina", "minQed", "minPocketFit", "searchText"].forEach((id) => {
    const el = byId(id);
    if (el) el.value = "";
  });
  applyFilters();
}

function renderCandTable() {
  const tb = byId("fullCandTableBody");
  if (!tb) return;
  if (!filteredData.length) {
    tb.innerHTML = `<tr><td colspan="18">No project candidates match the current filters.</td></tr>`;
    renderPagination();
    return;
  }

  const start = (currentPage - 1) * PAGE_SIZE;
  const page = filteredData.slice(start, start + PAGE_SIZE);
  tb.innerHTML = page.map((candidate) => {
    const idx = ALL.findIndex((row) => row.candidate_id === candidate.candidate_id);
    const globalIndex = idx >= 0 ? idx : 0;
    return `
      <tr onclick="openCandModal(${globalIndex})">
        <td>${escapeHtml(candidateRank(candidate, globalIndex))}</td>
        <td>${escapeHtml(candidateLabel(candidate))}</td>
        <td>${escapeHtml(fmtScore(candidate.best_score))}</td>
        <td>${escapeHtml(fmtLoose(candidate.pocket_electronic_fit_score, 3))}</td>
        <td>${escapeHtml(fmtLoose(candidate.electrostatic_score, 3))}</td>
        <td>${escapeHtml(fmtLoose(candidate.hbond_score, 3))}</td>
        <td>${escapeHtml(fmtLoose(candidate.hydrophobic_score, 3))}</td>
        <td>${escapeHtml(fmtLoose(candidate.pocket_guided_final_score, 4))}</td>
        <td>${escapeHtml(fmtLoose(candidate.mw, 1))}</td>
        <td>${escapeHtml(fmtLoose(candidate.logp, 2))}</td>
        <td>${escapeHtml(fmtLoose(candidate.qed, 3))}</td>
        <td>${escapeHtml(fmtLoose(candidate.sa_score, 2))}</td>
        <td>${escapeHtml(fmtLoose(candidate.admet_score, 3))}</td>
        <td>${escapeHtml(fmtInt(candidate.max_ring_size))}</td>
        <td>${candidate.has_macrocycle === true ? "yes" : candidate.has_macrocycle === false ? "no" : "not available"}</td>
        <td>${escapeHtml(fmtInt(candidate.lipinski_violations))}</td>
        <td>${tierBadge(candidate.selection_tier)}</td>
        <td><button class="btn-view" type="button" onclick="event.stopPropagation();goTo3D(${globalIndex})">3D</button></td>
      </tr>`;
  }).join("");
  renderPagination();
}

function renderPagination() {
  const el = byId("tablePagination");
  if (!el) return;
  const totalPages = Math.ceil(filteredData.length / PAGE_SIZE);
  if (totalPages <= 1) {
    el.innerHTML = "";
    return;
  }

  const buttons = [];
  const addButton = (page, label = page) => {
    buttons.push(`<button class="page-btn ${page === currentPage ? "active" : ""}" type="button" onclick="setPage(${page})">${label}</button>`);
  };
  const addDots = () => buttons.push(`<span style="color:var(--muted);padding:6px">...</span>`);

  if (currentPage > 1) addButton(currentPage - 1, "Prev");
  for (let page = 1; page <= totalPages; page += 1) {
    if (page <= 2 || page > totalPages - 2 || Math.abs(page - currentPage) <= 1) {
      addButton(page);
    } else if (page === 3 || page === totalPages - 2) {
      addDots();
    }
  }
  if (currentPage < totalPages) addButton(currentPage + 1, "Next");
  el.innerHTML = buttons.join("");
}

function setPage(page) {
  currentPage = page;
  renderCandTable();
}

function exportCSV() {
  const columns = [
    "rank",
    "candidate_id",
    "smiles",
    "best_score",
    "pocket_electronic_fit_score",
    "pocket_guided_final_score",
    "final_weighted_score",
    "mw",
    "logp",
    "qed",
    "sa_score",
    "admet_score",
    "lipinski_violations",
    "has_macrocycle",
    "max_ring_size",
    "selection_tier",
    "decision",
  ];
  const csvRows = [columns.join(",")];
  filteredData.forEach((candidate) => {
    csvRows.push(columns.map((column) => {
      const raw = candidate[column];
      if (raw === null || raw === undefined) return "";
      const text = String(raw);
      return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
    }).join(","));
  });
  const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `emd_v5_5_filtered_candidates_${filteredData.length}.csv`;
  document.body.appendChild(link);
  link.click();
  URL.revokeObjectURL(link.href);
  link.remove();
}

// Candidate modal
function openCandModal(index) {
  const candidate = ALL[index];
  if (!candidate) return;
  const residues = String(candidate.contact_residues || "")
    .split(";")
    .map((item) => item.trim())
    .filter(Boolean);
  const content = byId("candModalContent");
  if (!content) return;

  content.innerHTML = `
    <div class="modal-title">${escapeHtml(candidateLabel(candidate))} - Rank #${escapeHtml(candidateRank(candidate, index))}</div>
    <div class="modal-section">
      <div class="modal-section-title">SMILES from project output</div>
      <div class="smiles-box">${escapeHtml(candidate.smiles || "not available")}</div>
    </div>
    <div class="modal-section">
      <div class="modal-section-title">Ranking and docking</div>
      <div class="modal-grid">
        ${makeStat("Vina-GPU score", `${fmtScore(candidate.best_score)} kcal/mol`)}
        ${makeStat("Pocket fit", fmtLoose(candidate.pocket_electronic_fit_score, 4))}
        ${makeStat("Pocket-guided final", fmtLoose(candidate.pocket_guided_final_score, 4))}
        ${makeStat("Weighted final", fmtLoose(candidate.final_weighted_score, 4))}
        ${makeStat("Selection tier", tierText(candidate.selection_tier))}
        ${makeStat("Decision", tierText(candidate.decision))}
      </div>
    </div>
    <div class="modal-section">
      <div class="modal-section-title">Pocket-electronic components</div>
      <div class="modal-grid">
        ${makeStat("Electrostatic", fmtLoose(candidate.electrostatic_score, 4))}
        ${makeStat("H-bond", fmtLoose(candidate.hbond_score, 4))}
        ${makeStat("Hydrophobic", fmtLoose(candidate.hydrophobic_score, 4))}
        ${makeStat("Aromatic", fmtLoose(candidate.aromatic_score, 4))}
        ${makeStat("H-bond pairs", fmtInt(candidate.hbond_opportunity_pairs))}
        ${makeStat("Hydrophobic contacts", fmtInt(candidate.hydrophobic_contacts))}
        ${makeStat("Aromatic contacts", fmtInt(candidate.aromatic_contacts))}
        ${makeStat("Polar contacts", fmtInt(candidate.polar_contacts))}
      </div>
    </div>
    <div class="modal-section">
      <div class="modal-section-title">Molecular properties</div>
      <div class="modal-grid">
        ${makeStat("MW", fmtLoose(candidate.mw, 2))}
        ${makeStat("LogP", fmtLoose(candidate.logp, 2))}
        ${makeStat("TPSA", fmtLoose(candidate.tpsa, 1))}
        ${makeStat("HBD", fmtInt(candidate.hbd))}
        ${makeStat("HBA", fmtInt(candidate.hba))}
        ${makeStat("Rotatable bonds", fmtInt(candidate.rotatable_bonds))}
        ${makeStat("QED", fmtLoose(candidate.qed, 3))}
        ${makeStat("SA score", fmtLoose(candidate.sa_score, 2))}
        ${makeStat("ADMET proxy", fmtLoose(candidate.admet_score, 3))}
        ${makeStat("Synthesis proxy", fmtLoose(candidate.synthesis_score, 3))}
        ${makeStat("Safety proxy", fmtLoose(candidate.safety_proxy_score, 3))}
        ${makeStat("Novelty score", fmtLoose(candidate.novelty_score, 3))}
        ${makeStat("Max ring size", fmtInt(candidate.max_ring_size))}
        ${makeStat("Macrocycle", candidate.has_macrocycle === true ? "yes" : candidate.has_macrocycle === false ? "no" : "not available")}
        ${makeStat("Lipinski violations", fmtInt(candidate.lipinski_violations))}
        ${makeStat("Veber pass", candidate.veber_pass === true ? "yes" : candidate.veber_pass === false ? "no" : "not available")}
      </div>
    </div>
    <div class="modal-section">
      <div class="modal-section-title">SE(3) auxiliary geometry evidence</div>
      <div class="modal-grid">
        ${makeStat("Geometry loss", fmtLoose(candidate.se3_geometry_loss, 4))}
        ${makeStat("Geometry confidence", fmtLoose(candidate.se3_geometry_confidence, 4))}
        ${makeStat("Geometry status", candidate.se3_geometry_status || "not available")}
      </div>
    </div>
    <div class="modal-section">
      <div class="modal-section-title">Contact residues (${residues.length})</div>
      <div class="residue-grid">${residues.length ? residues.map((residue) => `<span class="residue-chip">${escapeHtml(residue)}</span>`).join("") : "not available"}</div>
    </div>
    <div class="modal-section">
      <div class="modal-section-title">Record metadata</div>
      <div class="modal-grid">
        ${makeStat("Source", candidate.source || "not available")}
        ${makeStat("Parent molecule", candidate.parent_mol_id || "not available")}
        ${makeStat("InChIKey", candidate.inchikey || "not available")}
        ${makeStat("Diversity cluster", candidate.diversity_cluster || "not available")}
        ${makeStat("Pose score", fmtLoose(candidate.pose_score, 3))}
        ${makeStat("Pose decision", candidate.pose_decision || "not available")}
      </div>
    </div>
    <div class="modal-section">
      <button class="btn-primary" type="button" onclick="goTo3D(${index});document.getElementById('candModal').classList.remove('active')">Open exported pose view</button>
    </div>`;
  byId("candModal")?.classList.add("active");
}

function closeCandModal(event) {
  if (!event || event.target === event.currentTarget) {
    byId("candModal")?.classList.remove("active");
  }
}

// 3D viewer
function setupViewerInterface() {
  const section = byId("viewer3d");
  if (!section) return;
  const badge = section.querySelector(".section-badge");
  const subtitle = section.querySelector(".section-subtitle");
  if (badge) badge.textContent = "3Dmol.js - protein + docked pose";
  if (subtitle) {
    subtitle.textContent = "Interactive visualization of the real JAK2 receptor and exported docked candidate poses. Contact residues and pocket context come from project outputs.";
  }

  const sidebar = section.querySelector(".viewer-sidebar");
  if (sidebar && sidebar.dataset.viewerReady !== "true") {
    sidebar.innerHTML = `
      <h3>Select Candidate</h3>
      <input type="text" id="molSidebarSearch" class="sidebar-search" placeholder="Search by ID, rank, residue, or SMILES..." oninput="filterMolList()">
      <div class="molecule-list" id="molList"></div>

      <div class="viewer-divider"></div>
      <h3>Display Settings</h3>
      <div class="control-row">
        <label>View Mode</label>
        <select id="molViewMode" onchange="render3D()">
          <option value="protein_pose">Protein + Docked Pose</option>
          <option value="pose_only">Docked Pose Only</option>
          <option value="protein_only">Protein Only</option>
        </select>
      </div>
      <div class="control-row">
        <label>Ligand Style</label>
        <select id="mol3dStyle" onchange="render3D()">
          <option value="stick">Stick</option>
          <option value="ballstick">Ball + Stick</option>
          <option value="sphere">Space-filling</option>
          <option value="line">Line</option>
          <option value="cross">Cross</option>
        </select>
      </div>
      <div class="control-row">
        <label>Bond Thickness</label>
        <select id="bondRadius" onchange="render3D()">
          <option value="0.14">Fine</option>
          <option value="0.20" selected>Clear</option>
          <option value="0.28">Bold</option>
        </select>
      </div>
      <div class="control-row">
        <label>Ligand Color</label>
        <select id="mol3dColor" onchange="render3D()">
          <option value="default">Element</option>
          <option value="spectrum">Spectrum</option>
          <option value="chain">Chain</option>
          <option value="cyanCarbon">Cyan Carbon</option>
          <option value="greenCarbon">Green Carbon</option>
        </select>
      </div>
      <div class="control-row">
        <label>Protein Style</label>
        <select id="protein3dStyle" onchange="render3D()">
          <option value="cartoon">Cartoon</option>
          <option value="stick">Stick</option>
          <option value="line">Line</option>
          <option value="surface">Surface Preview</option>
        </select>
      </div>
      <div class="control-row">
        <label>Protein Color</label>
        <select id="protein3dColor" onchange="render3D()">
          <option value="chain">Chain</option>
          <option value="spectrum">Spectrum</option>
          <option value="residue">Residue</option>
          <option value="white">White</option>
        </select>
      </div>
      <div class="control-row">
        <label>Background</label>
        <select id="mol3dBg" onchange="render3DBg()">
          <option value="0x0a0a1a">Dark</option>
          <option value="0x000000">Black</option>
          <option value="0xffffff">White</option>
          <option value="0x1a1a2e">Navy</option>
        </select>
      </div>
      <div class="control-row">
        <label>Surface</label>
        <select id="surfaceMode" onchange="render3D()">
          <option value="none">None</option>
          <option value="ligand">Ligand VDW Shell</option>
          <option value="pocket">Pocket Contact Surface</option>
          <option value="both">Ligand + Pocket</option>
        </select>
      </div>
      <div class="control-row">
        <label>Surface Opacity</label>
        <select id="surfaceOpacity" onchange="render3D()">
          <option value="0.12">Light</option>
          <option value="0.20" selected>Balanced</option>
          <option value="0.34">Strong</option>
        </select>
      </div>
      <div class="control-row">
        <label>Labels</label>
        <select id="labelMode" onchange="render3D()">
          <option value="none">None</option>
          <option value="key_atoms">Key Atoms</option>
          <option value="heavy_atoms">Heavy Atoms</option>
          <option value="all_atoms">All Atoms</option>
          <option value="bond_lengths">Bond Lengths</option>
          <option value="contacts">Contact Residues</option>
          <option value="atoms_contacts">Atoms + Contacts</option>
        </select>
      </div>
      <div class="control-row">
        <label>Max Labels</label>
        <select id="maxLabels" onchange="render3D()">
          <option value="20">20</option>
          <option value="40" selected>40</option>
          <option value="80">80</option>
          <option value="9999">All</option>
        </select>
      </div>
      <div class="control-row">
        <label>Highlight Contacts</label>
        <input type="checkbox" id="showContactResidues" class="toggle-check" checked onchange="render3D()">
      </div>
      <div class="control-row">
        <label>Show Grid Box</label>
        <input type="checkbox" id="showGridBox" class="toggle-check" onchange="render3D()">
      </div>
      <div class="control-row">
        <label>Show Pocket Overlay</label>
        <input type="checkbox" id="showPocketOverlay" class="toggle-check" onchange="render3D()">
      </div>
      <div class="control-row">
        <label>Spin</label>
        <input type="checkbox" id="mol3dSpin" class="toggle-check" onchange="toggleSpin3D()">
      </div>
      <div class="btn-group" style="margin-top:12px">
        <button class="btn-sm" onclick="reset3DView()">Reset</button>
        <button class="btn-sm" onclick="zoom3DFit()">Zoom Fit</button>
        <button class="btn-sm" onclick="snap3D()">Snapshot</button>
      </div>
      <div class="btn-group" style="margin-top:8px">
        <button class="btn-sm" onclick="loadProteinView()">Protein + Pose</button>
        <button class="btn-sm" onclick="setViewMode('pose_only')">Pose Only</button>
        <button class="btn-sm" onclick="setViewMode('protein_only')">Protein Only</button>
      </div>
      <div class="viewer-status-strip" id="viewerStatusStrip"></div>

      <div class="viewer-divider"></div>
      <h3>Project Lookup</h3>
      <textarea id="customSmiles" rows="3" placeholder="Paste a candidate ID or exact project SMILES..." style="width:100%;padding:10px;background:rgba(15,15,40,0.8);border:1px solid rgba(100,120,255,0.15);border-radius:8px;color:#e8eaf6;font-family:'JetBrains Mono',monospace;font-size:11px;resize:vertical;outline:none;margin-bottom:8px"></textarea>
      <button class="btn-primary" style="width:100%" onclick="loadCustomSmiles()">Open Project Candidate</button>`;
    sidebar.dataset.viewerReady = "true";
  }

  setViewerMessage("Select a candidate to open the receptor and docked pose view.");
}

function init3DViewer() {
  setupViewerInterface();
  const container = byId("mol3dViewer");
  if (!container || viewer3d) {
    buildMolList();
    return;
  }
  buildMolList();
  if (typeof $3Dmol === "undefined") {
    setViewerMessage("3Dmol.js was not loaded. Candidate data are still available in the tables.");
    return;
  }
  viewer3d = $3Dmol.createViewer(container, { backgroundColor: 0x0a0a1a, antialias: true });
  if (ALL.length) {
    loadMol3D(selectedCandidateIndex);
  }
}

function buildMolList(rows = ALL) {
  const el = byId("molList");
  if (!el) return;
  const visibleRows = rows.length ? rows : ALL;
  if (!visibleRows.length) {
    el.innerHTML = `<div class="mol-item"><span class="mol-item-name">No candidates exported</span></div>`;
    return;
  }
  el.innerHTML = visibleRows.map((candidate) => {
    const idx = ALL.findIndex((row) => row.candidate_id === candidate.candidate_id);
    const hasPose = Boolean(VIEWER_STATE.poseCache[candidate.candidate_id] || POSE_PATHS[candidate.candidate_id]);
    const isActive = idx === selectedCandidateIndex;
    const rank = candidateRank(candidate, idx >= 0 ? idx : 0);
    const residues = contactResidueSelections(candidate).length;
    return `
      <div class="mol-item ${isActive ? "active" : ""}" onclick="loadMol3D(${idx >= 0 ? idx : 0})">
        <div class="mol-item-main">
          <span class="mol-item-rank">#${escapeHtml(rank)}</span>
          <span class="mol-item-name">${escapeHtml(candidate.candidate_id)}</span>
        </div>
        <div class="mol-item-meta">
          <span>${escapeHtml(fmtScore(candidate.best_score))}</span>
          <span>${escapeHtml(fmtLoose(candidate.pocket_electronic_fit_score, 3))}</span>
          <span>${residues} contacts</span>
          <span class="${hasPose ? "pose-ok" : "pose-missing"}">${hasPose ? "pose" : "no pose"}</span>
        </div>
      </div>`;
  }).join("");
}

function filterMolList() {
  const query = String(byId("molSidebarSearch")?.value || "").trim().toLowerCase();
  const rows = ALL.filter((candidate) => {
    if (!query) return true;
    const text = [
      candidate.candidate_id,
      candidate.rank,
      candidate.smiles,
      candidate.inchikey,
      candidate.contact_residues,
      candidate.selection_tier,
    ].filter(Boolean).join(" ").toLowerCase();
    return text.includes(query);
  });
  buildMolList(rows);
}

function goTo3D(index) {
  selectedCandidateIndex = index;
  selectedCandidate = ALL[index] || null;
  showSection("viewer3d");
  init3DViewer();
  loadMol3D(index);
}

function setViewerMessage(message) {
  const overlay = byId("viewerOverlayMsg");
  if (overlay) {
    overlay.style.display = "flex";
    overlay.textContent = message;
  }
}

function hideViewerMessage() {
  const overlay = byId("viewerOverlayMsg");
  if (overlay) overlay.style.display = "none";
}

function setViewerStatus(chips) {
  const el = byId("viewerStatusStrip");
  if (!el) return;
  if (!chips.length) {
    el.innerHTML = "";
    return;
  }
  el.innerHTML = chips.map((chip) => `<span class="viewer-status-chip">${escapeHtml(chip)}</span>`).join("");
}

function normalizePoseText(rawText) {
  const lines = [];
  let inModel = false;
  String(rawText || "").split(/\r?\n/).forEach((line) => {
    const stripped = line.trim();
    if (stripped.startsWith("MODEL")) {
      inModel = /\b1\b/.test(stripped);
      return;
    }
    if (stripped.startsWith("ENDMDL") && inModel) return;
    if (stripped.startsWith("ATOM") || stripped.startsWith("HETATM")) {
      lines.push(stripped);
    }
  });
  return lines.length ? `${lines.join("\n")}\nEND\n` : "";
}

async function resolvePoseText(candidate) {
  const candidateId = candidate?.candidate_id;
  if (!candidateId) return null;
  if (VIEWER_STATE.poseCache[candidateId]) {
    VIEWER_STATE.poseSource = "embedded";
    return VIEWER_STATE.poseCache[candidateId];
  }
  const relativePath = POSE_PATHS[candidateId];
  if (!relativePath || typeof fetch === "undefined") {
    VIEWER_STATE.poseSource = relativePath ? "path_only" : "not_available";
    return null;
  }
  try {
    const response = await fetch(relativePath);
    if (!response.ok) {
      VIEWER_STATE.poseSource = "fetch_failed";
      return null;
    }
    const normalized = normalizePoseText(await response.text());
    if (!normalized) {
      VIEWER_STATE.poseSource = "empty_pose";
      return null;
    }
    VIEWER_STATE.poseCache[candidateId] = normalized;
    VIEWER_STATE.poseSource = "fetched";
    return normalized;
  } catch (error) {
    console.warn("Pose fetch failed", error);
    VIEWER_STATE.poseSource = "fetch_failed";
    return null;
  }
}

function parseResidueToken(token) {
  const parts = String(token || "").split(":");
  if (parts.length < 3) return null;
  const resi = numberValue(parts[2]);
  if (resi === null) return null;
  return {
    resn: parts[0],
    chain: parts[1],
    resi,
  };
}

function contactResidueSelections(candidate) {
  return String(candidate?.contact_residues || "")
    .split(";")
    .map((item) => parseResidueToken(item.trim()))
    .filter(Boolean);
}

function pocketResidueSelections() {
  const rows = Array.isArray(POCKET.top_contact_residues) ? POCKET.top_contact_residues : [];
  return rows
    .map((row) => parseResidueToken(row.residue))
    .filter(Boolean);
}

function groupedResidueSelection(selections) {
  const byChain = new Map();
  selections.forEach((selection) => {
    const key = selection.chain || "";
    if (!byChain.has(key)) byChain.set(key, []);
    byChain.get(key).push(selection.resi);
  });
  return [...byChain.entries()].map(([chain, residues]) => ({
    chain,
    resi: [...new Set(residues)],
  }));
}

function ligandStyleSpec() {
  const style = byId("mol3dStyle")?.value || "stick";
  const color = byId("mol3dColor")?.value || "default";
  const radius = numberValue(byId("bondRadius")?.value) ?? 0.2;
  const spec = {};
  if (style === "stick") spec.stick = { radius };
  if (style === "ballstick") {
    spec.stick = { radius: Math.max(radius * 0.75, 0.1) };
    spec.sphere = { scale: 0.22 };
  }
  if (style === "sphere") spec.sphere = { scale: 0.35 };
  if (style === "line") spec.line = { linewidth: 2 };
  if (style === "cross") spec.cross = { linewidth: 2 };

  const activeKeys = Object.keys(spec);
  if (!activeKeys.length) return { stick: { radius } };
  activeKeys.forEach((activeKey) => {
    if (color === "spectrum") spec[activeKey].colorscheme = "spectrum";
    if (color === "chain") spec[activeKey].colorscheme = "chainHetatm";
    if (color === "cyanCarbon") spec[activeKey].colorscheme = "cyanCarbon";
    if (color === "greenCarbon") spec[activeKey].colorscheme = "greenCarbon";
  });
  return spec;
}

function labelStyle(extra = {}) {
  return {
    font: "Arial",
    fontSize: 10,
    fontColor: "white",
    fontOpacity: 0.96,
    showBackground: true,
    backgroundColor: "#050510",
    backgroundOpacity: 0.55,
    borderThickness: 0.5,
    borderColor: "#22d3ee",
    inFront: false,
    ...extra,
  };
}

function ligandAtoms() {
  if (!VIEWER_STATE.ligandModel || typeof VIEWER_STATE.ligandModel.selectedAtoms !== "function") return [];
  try {
    return VIEWER_STATE.ligandModel.selectedAtoms({}) || [];
  } catch (error) {
    console.warn("Unable to read ligand atoms for labels", error);
    return [];
  }
}

function atomElement(atom) {
  const raw = String(atom?.elem || atom?.element || atom?.atom || "").replace(/[0-9]/g, "").trim();
  if (!raw) return "?";
  if (raw.length === 1) return raw.toUpperCase();
  return raw[0].toUpperCase() + raw.slice(1).toLowerCase();
}

function atomLabelText(atom, index) {
  const elem = atomElement(atom);
  return `${elem}${index + 1}`;
}

function atomLabelSelection(atoms, mode) {
  if (mode === "all_atoms") return atoms;
  if (mode === "heavy_atoms" || mode === "atoms_contacts") return atoms.filter((atom) => atomElement(atom) !== "H");
  return atoms.filter((atom) => !["C", "H"].includes(atomElement(atom)));
}

function addLigandAtomLabels(mode) {
  if (!viewer3d || !VIEWER_STATE.ligandModel) return;
  const maxLabels = numberValue(byId("maxLabels")?.value) ?? 40;
  const atoms = atomLabelSelection(ligandAtoms(), mode).slice(0, maxLabels);
  atoms.forEach((atom, index) => {
    viewer3d.addLabel(atomLabelText(atom, index), {
      ...labelStyle({ borderColor: "#34d399", backgroundOpacity: 0.42 }),
      position: { x: atom.x, y: atom.y, z: atom.z },
    });
  });
}

function addBondLengthLabels() {
  if (!viewer3d || !VIEWER_STATE.ligandModel) return;
  const maxLabels = numberValue(byId("maxLabels")?.value) ?? 40;
  const atoms = ligandAtoms();
  const byIndex = new Map(atoms.map((atom) => [atom.index, atom]));
  const seen = new Set();
  let added = 0;
  atoms.forEach((atom) => {
    if (!Array.isArray(atom.bonds)) return;
    atom.bonds.forEach((bondIndex) => {
      const other = byIndex.get(bondIndex);
      if (!other || added >= maxLabels) return;
      const key = [atom.index, other.index].sort((a, b) => a - b).join("-");
      if (seen.has(key)) return;
      seen.add(key);
      const dx = atom.x - other.x;
      const dy = atom.y - other.y;
      const dz = atom.z - other.z;
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (!Number.isFinite(distance) || distance <= 0) return;
      viewer3d.addLabel(`${distance.toFixed(2)} A`, {
        ...labelStyle({
          fontSize: 9,
          fontColor: "#d1fae5",
          borderColor: "#34d399",
          backgroundOpacity: 0.35,
        }),
        position: {
          x: (atom.x + other.x) / 2,
          y: (atom.y + other.y) / 2,
          z: (atom.z + other.z) / 2,
        },
      });
      added += 1;
    });
  });
}

function proteinStyleSpec() {
  const style = byId("protein3dStyle")?.value || "cartoon";
  const color = byId("protein3dColor")?.value || "chain";
  const colorSpec = color === "white"
    ? { color: "white" }
    : color === "residue"
      ? { colorscheme: "amino" }
      : color === "spectrum"
        ? { colorscheme: "spectrum" }
        : { colorscheme: "chain" };

  if (style === "stick") return { stick: { radius: 0.12, ...colorSpec } };
  if (style === "line") return { line: { linewidth: 1.2, ...colorSpec } };
  return { cartoon: { opacity: style === "surface" ? 0.45 : 0.95, ...colorSpec } };
}

function currentViewMode() {
  return byId("molViewMode")?.value || "protein_pose";
}

function applyResidueHighlights(candidate) {
  if (!VIEWER_STATE.proteinModel) return;
  const showContacts = Boolean(byId("showContactResidues")?.checked);
  const showPocket = Boolean(byId("showPocketOverlay")?.checked);
  if (showContacts) {
    contactResidueSelections(candidate).forEach((selection) => {
      VIEWER_STATE.proteinModel.setStyle(selection, {
        stick: { radius: 0.22, color: "#22d3ee" },
        cartoon: { color: "#22d3ee" },
      });
    });
  }
  if (showPocket) {
    pocketResidueSelections().forEach((selection) => {
      VIEWER_STATE.proteinModel.setStyle(selection, {
        stick: { radius: 0.16, color: "#fbbf24" },
      });
    });
  }
}

function applyLabels(candidate) {
  const labelMode = byId("labelMode")?.value || "none";
  if (labelMode === "none") return;
  if ((labelMode === "key_atoms" || labelMode === "heavy_atoms" || labelMode === "all_atoms" || labelMode === "atoms_contacts") && VIEWER_STATE.ligandModel) {
    addLigandAtomLabels(labelMode);
  }
  if (labelMode === "bond_lengths" && VIEWER_STATE.ligandModel) {
    addBondLengthLabels();
  }
  if ((labelMode === "contacts" || labelMode === "atoms_contacts") && VIEWER_STATE.proteinModel) {
    groupedResidueSelection(contactResidueSelections(candidate)).forEach((selection) => {
      viewer3d.addResLabels(selection, {
        ...labelStyle({
          fontSize: 12,
          borderColor: "#fbbf24",
          backgroundColor: "#0b0b20",
          backgroundOpacity: 0.62,
        }),
      });
    });
  }
}

function applySurfaces(candidate) {
  const surfaceMode = byId("surfaceMode")?.value || "none";
  const opacity = numberValue(byId("surfaceOpacity")?.value) ?? 0.2;
  const mode = currentViewMode();
  if ((surfaceMode === "ligand" || surfaceMode === "both") && VIEWER_STATE.ligandModel && mode !== "protein_only") {
    viewer3d.addSurface(
      $3Dmol.SurfaceType.VDW,
      { opacity, color: "#34d399" },
      { model: VIEWER_STATE.ligandModel }
    );
  }
  if ((surfaceMode === "pocket" || surfaceMode === "both") && VIEWER_STATE.proteinModel && mode !== "pose_only") {
    groupedResidueSelection(contactResidueSelections(candidate)).forEach((selection) => {
      viewer3d.addSurface(
        $3Dmol.SurfaceType.SAS || $3Dmol.SurfaceType.MS,
        { opacity: Math.min(opacity + 0.06, 0.42), color: "#fbbf24" },
        selection
      );
    });
  }
}

function applyGridBox() {
  if (!byId("showGridBox")?.checked) return;
  if (![GRID.center_x, GRID.center_y, GRID.center_z, GRID.size_x, GRID.size_y, GRID.size_z].every((value) => numberValue(value) !== null)) return;
  viewer3d.addBox({
    center: { x: GRID.center_x, y: GRID.center_y, z: GRID.center_z },
    dimensions: { w: GRID.size_x, h: GRID.size_y, d: GRID.size_z },
    color: "#34d399",
    opacity: 0.18,
    wireframe: true,
  });
}

function updateViewerStatus(candidate, hasPose) {
  const chips = [
    candidate ? candidate.candidate_id : "no candidate",
    `mode: ${currentViewMode()}`,
    `protein: ${PROTEIN_PDB ? "loaded" : "missing"}`,
    `pose: ${hasPose ? VIEWER_STATE.poseSource : "not available"}`,
    `contacts: ${contactResidueSelections(candidate).length}`,
    `grid box: ${byId("showGridBox")?.checked ? "on" : "off"}`,
  ];
  setViewerStatus(chips);
}

async function loadMol3D(index) {
  const candidate = ALL[index];
  if (!candidate) {
    setViewerMessage("No candidate was selected.");
    return;
  }
  selectedCandidateIndex = index;
  selectedCandidate = candidate;
  document.querySelectorAll(".mol-item").forEach((item) => {
    item.classList.toggle("active", item.textContent.includes(candidate.candidate_id));
  });
  setViewerMessage(`Loading ${candidate.candidate_id}...`);

  const poseText = await resolvePoseText(candidate);
  VIEWER_STATE.activePoseText = poseText;
  updateMolInfo(candidate);
  updateViewerStatus(candidate, Boolean(poseText));

  if (!viewer3d) {
    hideViewerMessage();
    return;
  }

  viewer3d.removeAllModels();
  viewer3d.removeAllLabels();
  viewer3d.removeAllShapes();
  viewer3d.removeAllSurfaces();
  VIEWER_STATE.proteinModel = null;
  VIEWER_STATE.ligandModel = null;

  try {
    if (PROTEIN_PDB) {
      VIEWER_STATE.proteinModel = viewer3d.addModel(PROTEIN_PDB, "pdb");
    }
    if (poseText) {
      VIEWER_STATE.ligandModel = viewer3d.addModel(poseText, "pdb");
    }
    if (!PROTEIN_PDB && !poseText) {
      setViewerMessage(`Neither receptor nor pose data are available for ${candidate.candidate_id}.`);
      viewer3d.render();
      return;
    }
    render3D();
    if (poseText || PROTEIN_PDB) {
      hideViewerMessage();
    }
    if (!poseText) {
      setViewerMessage(`Protein view is available, but no docked pose text is currently loaded for ${candidate.candidate_id}.`);
    }
  } catch (error) {
    console.warn("3D model load failed", error);
    setViewerMessage(`The viewer could not render ${candidate.candidate_id} from the exported files.`);
  }
}

function render3D() {
  if (!viewer3d || !selectedCandidate) return;
  render3DBg();
  viewer3d.removeAllLabels();
  viewer3d.removeAllShapes();
  viewer3d.removeAllSurfaces();

  const mode = currentViewMode();
  if (VIEWER_STATE.proteinModel) VIEWER_STATE.proteinModel.setStyle({}, {});
  if (VIEWER_STATE.ligandModel) VIEWER_STATE.ligandModel.setStyle({}, {});

  if (VIEWER_STATE.proteinModel && mode !== "pose_only") {
    VIEWER_STATE.proteinModel.setStyle({}, proteinStyleSpec());
    applyResidueHighlights(selectedCandidate);
  }
  if (VIEWER_STATE.ligandModel && mode !== "protein_only") {
    VIEWER_STATE.ligandModel.setStyle({}, ligandStyleSpec());
  }

  applySurfaces(selectedCandidate);
  applyGridBox();
  applyLabels(selectedCandidate);
  toggleSpin3D();
  viewer3d.zoomTo();
  viewer3d.render();
  updateViewerStatus(selectedCandidate, Boolean(VIEWER_STATE.activePoseText));
}

function render3DBg() {
  if (!viewer3d) return;
  const raw = byId("mol3dBg")?.value || "0x0a0a1a";
  viewer3d.setBackgroundColor(Number(raw));
}

function toggleSpin3D() {
  if (!viewer3d) return;
  viewer3d.spin(Boolean(byId("mol3dSpin")?.checked));
  viewer3d.render();
}

function reset3DView() {
  if (!viewer3d) return;
  viewer3d.zoomTo();
  viewer3d.render();
}

function zoom3DFit() {
  reset3DView();
}

function setViewMode(mode) {
  const select = byId("molViewMode");
  if (select) select.value = mode;
  render3D();
}

function loadProteinView() {
  setViewMode("protein_pose");
}

function snap3D() {
  if (!viewer3d) return;
  const uri = viewer3d.pngURI();
  const link = document.createElement("a");
  link.href = uri;
  link.download = `${selectedCandidate?.candidate_id || "emd_pose"}_snapshot.png`;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function loadCustomSmiles() {
  const query = String(byId("customSmiles")?.value || "").trim();
  if (!query) {
    setViewerMessage("Enter a candidate ID or exact SMILES from the project outputs.");
    return;
  }
  const lower = query.toLowerCase();
  const idx = ALL.findIndex((candidate) =>
    String(candidate.candidate_id || "").toLowerCase() === lower ||
    String(candidate.smiles || "") === query
  );
  if (idx < 0) {
    setViewerMessage("This ID or SMILES was not found in the exported project outputs.");
    return;
  }
  loadMol3D(idx);
}

function updateMolInfo(candidate) {
  const el = byId("molInfoStrip");
  if (!el) return;
  const poseAvailable = Boolean(VIEWER_STATE.poseCache[candidate.candidate_id] || POSE_PATHS[candidate.candidate_id]);
  el.innerHTML = [
    ["Candidate", candidate.candidate_id],
    ["Rank", candidateRank(candidate)],
    ["Vina", fmtScore(candidate.best_score)],
    ["Pocket Fit", fmtLoose(candidate.pocket_electronic_fit_score, 3)],
    ["Final Score", fmtLoose(candidate.pocket_guided_final_score, 4)],
    ["Pose Source", poseAvailable ? VIEWER_STATE.poseSource.replace(/_/g, " ") : "not available"],
    ["Contact Residues", contactResidueSelections(candidate).length],
    ["Protein", PROTEIN_PDB ? "JAK2 receptor" : "not available"],
    ["MW", fmtLoose(candidate.mw, 1)],
    ["QED", fmtLoose(candidate.qed, 3)],
  ].map(([label, value]) => `
    <div class="mol-info-item">
      <div class="mol-info-label">${escapeHtml(label)}</div>
      <div class="mol-info-value">${escapeHtml(String(value))}</div>
    </div>`).join("");
}

// Docking and pocket section
function initDockingSection() {
  populatePocketStats();
  populateResidues();
  renderDockingCharts();
}

function populatePocketStats() {
  const engine = GRID.engine || "not available";
  const receptor = GRID.receptor || "not available";
  const gridSize = [GRID.size_x, GRID.size_y, GRID.size_z].map((value) => fmtLoose(value, 1)).join(" x ");
  const gridCenter = [GRID.center_x, GRID.center_y, GRID.center_z].map((value) => fmtLoose(value, 3)).join(", ");
  const profileResidues = Array.isArray(POCKET.top_contact_residues) ? POCKET.top_contact_residues.length : null;

  setText(
    "dockingSubtitle",
    `Real Vina-GPU docking outputs: ${fmtInt(S.docked_candidates)} parsed scores from ${fmtInt(S.raw_pose_files)} raw pose files. Grid center ${gridCenter}; grid size ${gridSize}.`
  );
  setText("pocketSourceBadge", "Loaded from docking scores and pocket profile JSON");
  setHtml("pocketStatGrid", [
    makePocketStat("PDB ID / receptor", receptor),
    makePocketStat("Docking engine", engine),
    makePocketStat("Grid center", gridCenter),
    makePocketStat("Grid size", gridSize),
    makePocketStat("Generated candidates", fmtInt(S.generated_candidates)),
    makePocketStat("Ranked candidates", fmtInt(S.ranked_candidates)),
    makePocketStat("Raw attempts", fmtInt(S.raw_attempts)),
    makePocketStat("Raw-attempt validity", fmtPct(S.raw_attempt_validity_percent, 4)),
    makePocketStat("Linker novelty", fmtPct(S.linker_novelty_percent, 2)),
    makePocketStat("Uniqueness", fmtPct(S.uniqueness_percent, 2)),
    makePocketStat("Macrocyclization", fmtPct(S.macrocyclization_percent, 2)),
    makePocketStat("Docked candidates", fmtInt(S.docked_candidates)),
    makePocketStat("Raw pose files", fmtInt(S.raw_pose_files)),
    makePocketStat("Pocket-scored rows", fmtInt(S.pocket_scored_candidates)),
    makePocketStat("Best Vina-GPU", `${fmtScore(S.best_vina_gpu_score)} kcal/mol`),
    makePocketStat("Median Vina-GPU", `${fmtScore(S.median_vina_gpu_score)} kcal/mol`),
    makePocketStat("Best pocket fit", fmtLoose(S.best_pocket_electronic_fit_score, 5)),
    makePocketStat("Median pocket fit", fmtLoose(S.median_pocket_electronic_fit_score, 5)),
    makePocketStat("Pocket atoms", fmtInt(POCKET.pocket_atom_count)),
    makePocketStat("Hydrophobic atoms", fmtInt(POCKET.hydrophobic_atom_count)),
    makePocketStat("H-bond donors", fmtInt(POCKET.hbond_donor_atom_count)),
    makePocketStat("H-bond acceptors", fmtInt(POCKET.hbond_acceptor_atom_count)),
    makePocketStat("Aromatic atoms", fmtInt(POCKET.aromatic_atom_count)),
    makePocketStat("Charge proxy", fmtLoose(POCKET.pocket_net_charge_proxy, 3)),
    makePocketStat("Top residue records", fmtInt(profileResidues)),
  ].join(""));
}

function populateResidues() {
  const top = ALL[0];
  const residues = String(top?.contact_residues || "")
    .split(";")
    .map((item) => item.trim())
    .filter(Boolean);
  setText("residueHeader", top ? `Key Binding Residues - ${top.candidate_id} (Rank #${candidateRank(top)})` : "Key Binding Residues");
  setText("residueBadge", residues.length ? `${residues.length} contact residues from ranked output` : "No residue list exported");
  const grid = byId("residueGrid");
  if (!grid) return;
  grid.innerHTML = residues.length
    ? residues.map((residue) => `<span class="residue-chip">${escapeHtml(residue)}</span>`).join("")
    : "not available";
}

// ADMET
function initAdmetSection() {
  renderAdmetSummary();
  renderAdmetCharts();
}

function renderAdmetSummary() {
  setText("admetBadge", `${ALL.length.toLocaleString()} ranked candidates`);
  const el = byId("admetSummary");
  if (!el) return;

  const rows = [
    ["Lipinski 0 violations", ADMET.lipinski_0, ALL.length],
    ["Lipinski 1 violation", ADMET.lipinski_1, ALL.length],
    ["Lipinski 2+ violations", ADMET.lipinski_2plus, ALL.length],
    ["Veber pass", ADMET.veber_pass, ALL.length],
    ["Veber fail", ADMET.veber_fail, ALL.length],
    ["Docked candidates", S.docked_candidates, ALL.length],
    ["Pocket-scored candidates", S.pocket_scored_candidates, ALL.length],
    ["Embedded pose rows", availablePoseCount(), ALL.length],
  ];

  el.innerHTML = rows.map(([name, value, total]) => {
    const v = numberValue(value);
    const t = numberValue(total);
    const pct = v !== null && t ? Math.max(0, Math.min(100, (v / t) * 100)) : null;
    return `
      <div class="admet-stat-row">
        <div class="admet-stat-name">${escapeHtml(name)}</div>
        <div class="admet-bar-track"><div class="admet-bar-fill" style="width:${pct ?? 0}%"></div></div>
        <div class="admet-stat-vals"><span>${escapeHtml(fmtInt(value))}</span><span>${pct === null ? "not available" : fmtPct(pct, 1)}</span></div>
      </div>`;
  }).join("");
}

// Pipeline
function initPipelineSection() {
  renderPipelineFlow();
  renderGeneratorStats();
  renderPipelineCharts();
}

function renderPipelineFlow() {
  const stages = [
    ["Data", fmtInt(S.curated_jak2_ligands), "Curated ligands"],
    ["Features", `${fmtInt(S.validation?.passed)}/${fmtInt(S.validation?.total)}`, "Validation checks"],
    ["V5.5 Generate", fmtInt(S.generated_candidates), "Generated candidates"],
    ["Attempt Log", fmtInt(S.raw_attempts), "Raw attempts"],
    ["Docking", fmtInt(S.docked_candidates), "Parsed Vina scores"],
    ["Pocket Fit", fmtInt(S.pocket_scored_candidates), "Scored rows"],
    ["ADMET", fmtInt(ALL.length), "Descriptor rows"],
    ["Ranking", candidateLabel(ALL[0]), "Top candidate"],
    ["Validation", `${fmtInt(S.validation?.passed)}/${fmtInt(S.validation?.total)}`, "Project checks"],
  ];
  const el = byId("pipelineFlow");
  if (!el) return;
  el.innerHTML = stages.map(([name, detail, caption], index) => `
    <div class="pipe-stage done">
      <div class="pipe-icon">${index + 1}</div>
      <div class="pipe-name">${escapeHtml(name)}</div>
      <div class="pipe-detail">${escapeHtml(detail)}<br>${escapeHtml(caption)}</div>
    </div>${index < stages.length - 1 ? '<div class="pipe-arrow">-&gt;</div>' : ""}`).join("");
}

function renderGeneratorStats() {
  setText("generatorStatsBadge", `From ${escapeHtml(S.active_branch_for_benchmark || "exported benchmark summary")}`);
  const tb = byId("generatorStatsBody") || document.querySelector("#pipeline .data-table tbody");
  if (!tb) return;
  if (!BRANCHES.length) {
    tb.innerHTML = `<tr><td colspan="10">No branch metrics were exported.</td></tr>`;
    return;
  }
  tb.innerHTML = BRANCHES.map((row) => `
    <tr>
      <td>${escapeHtml(row.branch || "not available")}</td>
      <td>${escapeHtml(fmtInt(row.num_candidates))}</td>
      <td>${escapeHtml(fmtInt(row.num_unique_inchikeys))}</td>
      <td>${escapeHtml(fmtPct(row.novel_molecule_percent, 2))}</td>
      <td>${escapeHtml(fmtPct(row.basic_filter_pass_percent, 2))}</td>
      <td>${escapeHtml(fmtPct(row.macrocyclization_percent, 2))}</td>
      <td>${escapeHtml(fmtLoose(row.median_mw, 1))}</td>
      <td>${escapeHtml(fmtLoose(row.median_logp, 2))}</td>
      <td>${escapeHtml(fmtLoose(row.median_qed, 3))}</td>
      <td>${escapeHtml(fmtLoose(row.median_sa_score, 3))}</td>
    </tr>`).join("");

  const card = tb.closest(".glass-card");
  if (!card || byId("attemptStatusGrid")) return;
  const attempts = Object.entries(ATTEMPTS);
  if (!attempts.length) return;
  const block = document.createElement("div");
  block.id = "attemptStatusGrid";
  block.style.marginTop = "18px";
  block.innerHTML = `
    <div class="card-header" style="margin-top:8px">
      <h2>Generation Attempt Status Counts</h2>
      <div class="card-badge">From v5_5_pocket_guided_summary.json</div>
    </div>
    <div class="pocket-stat-grid">
      ${attempts.map(([status, count]) => makePocketStat(status.replace(/_/g, " "), fmtInt(count))).join("")}
    </div>`;
  card.appendChild(block);
}

// Chart helpers
function chartReady(canvasId) {
  const canvas = byId(canvasId);
  if (!canvas) return null;
  if (typeof Chart === "undefined") {
    const fallback = document.createElement("div");
    fallback.className = "viewer-overlay-text";
    fallback.style.position = "relative";
    fallback.textContent = "Chart.js was not loaded. Numeric data remain available in the dashboard tables.";
    canvas.replaceWith(fallback);
    return null;
  }
  if (CHARTS[canvasId]) CHARTS[canvasId].destroy();
  return canvas.getContext("2d");
}

function renderChart(canvasId, config) {
  const ctx = chartReady(canvasId);
  if (!ctx) return;
  CHARTS[canvasId] = new Chart(ctx, config);
}

function numericArray(key) {
  return ALL.map((candidate) => numberValue(candidate[key])).filter((value) => value !== null);
}

function histogramRows(values, binWidth) {
  if (!values.length) return [];
  const buckets = new Map();
  values.forEach((value) => {
    const start = Math.floor(value / binWidth) * binWidth;
    buckets.set(start, (buckets.get(start) || 0) + 1);
  });
  return [...buckets.entries()].sort((a, b) => a[0] - b[0]).map(([start, count]) => ({
    label: `${start.toFixed(binWidth < 1 ? 2 : 1)} to ${(start + binWidth).toFixed(binWidth < 1 ? 2 : 1)}`,
    count,
  }));
}

function chartOptions(title = "") {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#e8eaf6" } },
      title: title ? { display: true, text: title, color: "#e8eaf6" } : undefined,
    },
    scales: {
      x: { ticks: { color: "#9fa4c4" }, grid: { color: "rgba(255,255,255,0.05)" } },
      y: { ticks: { color: "#9fa4c4" }, grid: { color: "rgba(255,255,255,0.05)" } },
    },
  };
}

function renderDockingCharts() {
  const dockingLabels = DOCKING_HIST.map((row) => row.bin);
  const dockingCounts = DOCKING_HIST.map((row) => row.count);
  renderChart("dockingHistChart", {
    type: "bar",
    data: { labels: dockingLabels, datasets: [{ label: "Docked candidates", data: dockingCounts, backgroundColor: "#22d3ee" }] },
    options: chartOptions(),
  });

  const scatterRows = ALL.filter((candidate) => numberValue(candidate.best_score) !== null && numberValue(candidate.pocket_electronic_fit_score) !== null);
  renderChart("scatterDockChart", {
    type: "scatter",
    data: {
      datasets: [{
        label: "Candidates with Vina and pocket scores",
        data: scatterRows.map((candidate) => ({ x: candidate.best_score, y: candidate.pocket_electronic_fit_score })),
        backgroundColor: "rgba(99,102,241,0.75)",
      }],
    },
    options: chartOptions(),
  });

  const componentRows = TOP.slice(0, 20);
  const componentAverages = ["electrostatic_score", "hbond_score", "hydrophobic_score", "aromatic_score"].map((key) => {
    const values = componentRows.map((candidate) => numberValue(candidate[key])).filter((value) => value !== null);
    return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  });
  renderChart("radarPocketChart", {
    type: "radar",
    data: {
      labels: ["Electrostatic", "H-bond", "Hydrophobic", "Aromatic"],
      datasets: [{ label: "Top 20 mean", data: componentAverages, borderColor: "#22d3ee", backgroundColor: "rgba(34,211,238,0.18)" }],
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: "#e8eaf6" } } }, scales: { r: { ticks: { color: "#9fa4c4" }, grid: { color: "rgba(255,255,255,0.08)" }, angleLines: { color: "rgba(255,255,255,0.08)" } } } },
  });

  const ringMap = new Map();
  ALL.forEach((candidate) => {
    const ring = numberValue(candidate.max_ring_size);
    const score = numberValue(candidate.best_score);
    if (ring === null || score === null) return;
    if (!ringMap.has(ring)) ringMap.set(ring, []);
    ringMap.get(ring).push(score);
  });
  const ringLabels = [...ringMap.keys()].sort((a, b) => a - b);
  const ringData = ringLabels.map((ring) => {
    const values = ringMap.get(ring);
    return values.reduce((a, b) => a + b, 0) / values.length;
  });
  renderChart("ringSizeChart", {
    type: "bar",
    data: { labels: ringLabels.map(String), datasets: [{ label: "Mean Vina score", data: ringData, backgroundColor: "#34d399" }] },
    options: chartOptions(),
  });
}

function renderAdmetCharts() {
  const specs = [
    ["qedHistChart", "qed", 0.05, "QED"],
    ["mwHistChart", "mw", 50, "Molecular weight"],
    ["logpHistChart", "logp", 0.5, "LogP"],
    ["saHistChart", "sa_score", 0.25, "SA score"],
  ];
  specs.forEach(([canvasId, key, width, label]) => {
    const rows = histogramRows(numericArray(key), width);
    renderChart(canvasId, {
      type: "bar",
      data: { labels: rows.map((row) => row.label), datasets: [{ label, data: rows.map((row) => row.count), backgroundColor: "#818cf8" }] },
      options: chartOptions(),
    });
  });

  const scatterRows = ALL.filter((candidate) => numberValue(candidate.qed) !== null && numberValue(candidate.admet_score) !== null);
  renderChart("admetQedScatter", {
    type: "scatter",
    data: {
      datasets: [{
        label: "Candidates",
        data: scatterRows.map((candidate) => ({ x: candidate.qed, y: candidate.admet_score })),
        backgroundColor: scatterRows.map((candidate) => {
          const violations = numberValue(candidate.lipinski_violations) || 0;
          if (violations === 0) return "rgba(52,211,153,0.75)";
          if (violations === 1) return "rgba(251,191,36,0.75)";
          return "rgba(251,113,133,0.75)";
        }),
      }],
    },
    options: chartOptions(),
  });
}

function renderPipelineCharts() {
  renderChart("genPieChart", {
    type: "doughnut",
    data: {
      labels: BRANCHES.map((row) => row.branch || "not available"),
      datasets: [{ label: "Candidates", data: BRANCHES.map((row) => numberValue(row.num_candidates) || 0), backgroundColor: ["#22d3ee", "#818cf8", "#34d399", "#fbbf24", "#fb7185", "#a78bfa"] }],
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: "#e8eaf6" } } } },
  });

  const rows = TOP.slice(0, 10);
  renderChart("scoringRadar", {
    type: "bar",
    data: {
      labels: rows.map((candidate) => candidate.candidate_id),
      datasets: [
        { label: "Final", data: rows.map((candidate) => numberValue(candidate.pocket_guided_final_score)), backgroundColor: "#22d3ee" },
        { label: "Pocket fit", data: rows.map((candidate) => numberValue(candidate.pocket_electronic_fit_score)), backgroundColor: "#818cf8" },
        { label: "ADMET", data: rows.map((candidate) => numberValue(candidate.admet_score)), backgroundColor: "#34d399" },
      ],
    },
    options: chartOptions(),
  });

  const ringCounts = new Map();
  ALL.forEach((candidate) => {
    const ring = numberValue(candidate.max_ring_size);
    if (ring === null) return;
    ringCounts.set(ring, (ringCounts.get(ring) || 0) + 1);
  });
  const labels = [...ringCounts.keys()].sort((a, b) => a - b);
  renderChart("ringSizeBarChart", {
    type: "bar",
    data: { labels: labels.map(String), datasets: [{ label: "Candidate count", data: labels.map((ring) => ringCounts.get(ring)), backgroundColor: "#fbbf24" }] },
    options: chartOptions(),
  });
}

function initializeDashboard() {
  setupViewerInterface();
  populateHeaderAndKpis();
  buildHeroCandidates();
  buildOverviewTable();
  applyFilters();
  initDockingSection();
  initAdmetSection();
  initPipelineSection();

  const disclaimer = document.querySelector(".footer-disclaimer");
  if (disclaimer) disclaimer.textContent = "Computational candidates only. Not experimentally validated.";

  if (!ALL.length) {
    console.warn("No candidate data was exported to data.js.");
  }
  console.info(truthNote());
}

document.addEventListener("DOMContentLoaded", initializeDashboard);
