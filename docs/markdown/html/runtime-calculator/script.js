// Constants
const PC_TO_KM = 3.08567758e13;
const MYR_TO_S = 3.15576e13; // 1 Myr = 1e6 * 365.25 * 86400
const SEC_TO_YR = 1 / (365.25 * 86400);

// State
let levels = [ {id : 0, nx : 128, ny : 128, nz : 128, filling : 100} ];

// DOM Elements
const levelsContainer = document.getElementById('levels-container');
const addLevelBtn = document.getElementById('add-level-btn');
const manualStepsToggle = document.getElementById('manual-steps-toggle');
const autoStepsInputs = document.getElementById('auto-steps-inputs');
const manualStepsInputs = document.getElementById('manual-steps-inputs');
const themeToggle = document.getElementById('theme-toggle');
const themeIcon = document.querySelector('.theme-icon');

// Inputs
const inputs = {
  gpuPerf : document.getElementById('gpu-perf'),
  perfUnit : document.getElementById('perf-unit'),
  boxSize : document.getElementById('box-size'),
  vmax : document.getElementById('vmax'),
  cfl : document.getElementById('cfl'),
  tStop : document.getElementById('t-stop'),
  tStopUnit : document.getElementById('t-stop-unit'),
  maxSteps : document.getElementById('max-steps'),
  numGpus : document.getElementById('num-gpus'),
  suConversion : document.getElementById('su-conversion')
};

// Outputs
const outputs = {
  totalCells : document.getElementById('total-cells'),
  minDx : document.getElementById('min-dx'),
  dt : document.getElementById('dt-val'),
  totalSteps : document.getElementById('total-steps'),
  rtDays : document.getElementById('rt-days'),
  rtHours : document.getElementById('rt-hours'),
  rtSeconds : document.getElementById('rt-seconds'),
  gpuHours : document.getElementById('gpu-hours'),
  suHours : document.getElementById('su-hours')
};

// Initialization
function init() {
  renderLevels();
  addEventListeners();
  initTheme();
  calculate();
}

function initTheme() {
  // Default to light mode (no dark-mode class)
  // Check if user has a saved preference
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme === 'dark') {
    document.body.classList.add('dark-mode');
    themeIcon.textContent = '☀️';
  } else {
    themeIcon.textContent = '🌙';
  }
}

function addEventListeners() {
  addLevelBtn.addEventListener('click', addLevel);

  // Theme toggle
  themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    themeIcon.textContent = isDark ? '☀️' : '🌙';
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  });

  // Toggle between manual and auto steps
  manualStepsToggle.addEventListener('change', (e) => {
    if (e.target.checked) {
      autoStepsInputs.classList.add('hidden');
      manualStepsInputs.classList.remove('hidden');
    } else {
      autoStepsInputs.classList.remove('hidden');
      manualStepsInputs.classList.add('hidden');
    }
    calculate();
  });

  // Attach listeners to all static inputs
  Object.values(inputs).forEach(
      input => { input.addEventListener('input', calculate); });

  // Export button
  document.getElementById('export-btn')
      .addEventListener('click', exportCalculation);
}

// Level Management
function renderLevels() {
  levelsContainer.innerHTML = '';
  levels.forEach((level, index) => {
    const div = document.createElement('div');
    div.className = 'level-row';
    div.innerHTML = `
            <div class="level-header">
                <span>Level ${index}</span>
                ${
        index > 0 ? `<button class="remove-level-btn" onclick="removeLevel(${
                        index})">×</button>`
                  : ''}
            </div>
            <div class="level-inputs">
                <div>
                    <label>Nx</label>
                    <input type="number" value="${
        level.nx}" step="64" oninput="updateLevel(${index}, 'nx', this.value)">
                </div>
                <div>
                    <label>Ny</label>
                    <input type="number" value="${
        level.ny}" step="64" oninput="updateLevel(${index}, 'ny', this.value)">
                </div>
                <div>
                    <label>Nz</label>
                    <input type="number" value="${
        level.nz}" step="64" oninput="updateLevel(${index}, 'nz', this.value)">
                </div>
                <div>
                    <label>Fill %
                        <span class="help-icon" data-tooltip="Percentage of cells that are refined at this level (0-100%). Default is 100% if left empty.">ℹ️</span>
                    </label>
                    <input type="number" value="${
        level.filling}" placeholder="100" oninput="updateLevel(${
        index}, 'filling', this.value)">
                </div>
            </div>
        `;
    levelsContainer.appendChild(div);
  });
}

function addLevel() {
  const lastLevel = levels[levels.length - 1];
  levels.push({
    id : levels.length,
    nx : lastLevel.nx * 2,
    ny : lastLevel.ny * 2,
    nz : lastLevel.nz * 2,
    filling : 10 // Default to small filling for high levels
  });
  renderLevels();
  calculate();
}

// Global scope for HTML access
window.removeLevel = function(index) {
  if (levels.length <= 1)
    return;
  levels.splice(index, 1);
  renderLevels();
  calculate();
};

window.updateLevel = function(index, field, value) {
  levels[index][field] = parseFloat(value) || 0;
  calculate();
};

// Calculation Logic
function calculate() {
  // 1. Calculate Total Cells
  let totalCells = 0;
  let maxNx = 0;

  levels.forEach(l => {
    let fill = l.filling;
    if (isNaN(fill))
      fill = 100;

    const cells = l.nx * l.ny * l.nz * (fill / 100);
    totalCells += cells;

    if (l.nx > maxNx)
      maxNx = l.nx;
  });

  // 2. Physics & Time
  const boxSizePc = parseFloat(inputs.boxSize.value) || 0;
  const vmax = parseFloat(inputs.vmax.value) || 0;
  const cfl = parseFloat(inputs.cfl.value) || 0;
  const numGpus = parseFloat(inputs.numGpus.value) || 1;

  // Convert performance to GPU-ns
  const perfValue = parseFloat(inputs.gpuPerf.value) || 0;
  const perfUnit = inputs.perfUnit.value;
  let gpuNs = 0;

  if (perfUnit === 'ns') {
    gpuNs = perfValue;
  } else if (perfUnit === 'mupdates') {
    // x Mupdates/(GPU*s) = 1000/x GPU*ns/update
    gpuNs = perfValue > 0 ? 1000 / perfValue : 0;
  }

  // Derived: min dx
  let minDxPc = 0;
  let minDxKm = 0;

  // Find max level index
  const maxLevelIndex = levels.length - 1;
  const level0 = levels[0];

  if (boxSizePc > 0 && level0.nx > 0) {
    // dx at level 0
    const dx0 = boxSizePc / level0.nx;
    // dx at finest level = dx0 / 2^maxLevelIndex
    minDxPc = dx0 / Math.pow(2, maxLevelIndex);
    minDxKm = minDxPc * PC_TO_KM;
  }

  // Derived: dt
  let dt = 0;
  if (vmax > 0 && minDxKm > 0 && cfl > 0) {
    dt = cfl * minDxKm / vmax;
  }

  // Derived: Total Steps
  let totalSteps = 0;
  const isManualSteps = manualStepsToggle.checked;

  if (isManualSteps) {
    totalSteps = parseFloat(inputs.maxSteps.value) || 0;
  } else {
    const tStopVal = parseFloat(inputs.tStop.value) || 0;
    const unit = inputs.tStopUnit.value;
    let tStopS = 0;

    if (unit === 'Myr')
      tStopS = tStopVal * MYR_TO_S;
    else
      tStopS = tStopVal; // seconds

    if (dt > 0) {
      totalSteps = tStopS / dt;
    }
  }

  // 3. Runtime & Cost
  const totalUpdates = totalCells * totalSteps;

  // Runtime (s)
  let runtimeS = 0;
  if (gpuNs > 0 && numGpus > 0) {
    runtimeS = totalUpdates * (gpuNs * 1e-9) / numGpus;
  }

  // 4. Update UI
  outputs.totalCells.textContent = formatNumber(totalCells);
  outputs.minDx.textContent =
      minDxPc > 0 ? `${minDxPc.toExponential(2)} pc` : '-';

  // Display dt in Years
  const dtYears = dt * SEC_TO_YR;
  outputs.dt.textContent = dt > 0 ? `${dtYears.toExponential(2)} yr` : '-';

  outputs.totalSteps.textContent = formatNumber(Math.floor(totalSteps));

  // Format Runtime (Equivalent quantities)
  const totalDays = runtimeS / 86400;
  const totalHours = runtimeS / 3600;

  outputs.rtDays.textContent = formatNumber(totalDays);
  outputs.rtHours.textContent = formatNumber(totalHours);
  outputs.rtSeconds.textContent =
      runtimeS > 0 ? runtimeS.toExponential(2) : '0';

  // Cost
  const cost = (runtimeS / 3600) * numGpus;
  outputs.gpuHours.textContent = formatNumber(cost);

  // SU Cost
  const suConversion = parseFloat(inputs.suConversion.value) || 64;
  const suCost = cost * suConversion;
  outputs.suHours.textContent = formatNumber(suCost);
}

function formatNumber(num) {
  if (num === 0)
    return '0';
  if (num >= 1e6 || (num < 0.01 && num > 0))
    return num.toExponential(2);
  return num.toLocaleString(undefined, {maximumFractionDigits : 2});
}

// Start
init();

function exportCalculation() {
  // Gather Inputs
  const exportData = {
    timestamp : new Date().toISOString(),
    inputs : {
      hardware : {
        gpuPerformance : parseFloat(inputs.gpuPerf.value) || 0,
        performanceUnit : inputs.perfUnit.value,
        numGpus : parseFloat(inputs.numGpus.value) || 1,
        suPerGpuHour : parseFloat(inputs.suConversion.value) || 64
      },
      physics : {
        boxSizePc : parseFloat(inputs.boxSize.value) || 0,
        vmaxKms : parseFloat(inputs.vmax.value) || 0,
        cfl : parseFloat(inputs.cfl.value) || 0,
        tStop : parseFloat(inputs.tStop.value) || 0,
        tStopUnit : inputs.tStopUnit.value
      },
      grid : levels
    },
    results : {}
  };

  // Gather Results (Recalculate to ensure precision, or parse from DOM)
  // Parsing from DOM is safer to match what user sees, but less precise.
  // Let's recalculate briefly or grab from the DOM text and clean it.
  // Actually, let's just grab the text content for simplicity and "what you see
  // is what you get", but for the specific fields requested (runtimeHours, etc)
  // we might need to derive them if they aren't explicitly on screen in that
  // format. Wait, the user requested specific fields: runtimeHours,
  // runtimeDays. These are calculated in `calculate()` but not all stored in
  // global state. I should probably refactor `calculate` to return values or
  // store them, OR just re-implement the calculation logic inside export
  // (duplication bad), OR just parse the display values where possible and do
  // simple math.

  // Let's try to extract the raw values from the DOM elements where possible,
  // or re-derive simple ones.

  // Re-deriving is safer for the "runtimeHours" etc requirements.

  // 1. Total Cells
  let totalCells = 0;
  let maxNx = 0;
  levels.forEach(l => {
    let fill = l.filling;
    if (isNaN(fill))
      fill = 100;
    const cells = l.nx * l.ny * l.nz * (fill / 100);
    totalCells += cells;
    if (l.nx > maxNx)
      maxNx = l.nx;
  });

  // 2. Physics
  const boxSizePc = parseFloat(inputs.boxSize.value) || 0;
  const vmax = parseFloat(inputs.vmax.value) || 0;
  const cfl = parseFloat(inputs.cfl.value) || 0;
  const numGpus = parseFloat(inputs.numGpus.value) || 1;

  // Perf
  const perfValue = parseFloat(inputs.gpuPerf.value) || 0;
  const perfUnit = inputs.perfUnit.value;
  let gpuNs = 0;
  if (perfUnit === 'ns')
    gpuNs = perfValue;
  else if (perfUnit === 'mupdates')
    gpuNs = perfValue > 0 ? 1000 / perfValue : 0;

  // Derived
  let minDxPc = 0;
  if (boxSizePc > 0 && maxNx > 0)
    minDxPc = boxSizePc / maxNx;
  const minDxKm = minDxPc * PC_TO_KM;

  let dt = 0;
  if (vmax > 0 && minDxKm > 0 && cfl > 0)
    dt = cfl * minDxKm / vmax;
  const dtYears = dt * SEC_TO_YR;

  let totalSteps = 0;
  if (manualStepsToggle.checked) {
    totalSteps = parseFloat(inputs.maxSteps.value) || 0;
  } else {
    const tStopVal = parseFloat(inputs.tStop.value) || 0;
    const unit = inputs.tStopUnit.value;
    let tStopS = unit === 'Myr' ? tStopVal * MYR_TO_S : tStopVal;
    if (dt > 0)
      totalSteps = tStopS / dt;
  }

  const totalUpdates = totalCells * totalSteps;
  let runtimeS = 0;
  if (gpuNs > 0 && numGpus > 0)
    runtimeS = totalUpdates * (gpuNs * 1e-9) / numGpus;

  const cost = (runtimeS / 3600) * numGpus;
  const suConversion = parseFloat(inputs.suConversion.value) || 64;
  const suCost = cost * suConversion;

  exportData.results = {
    totalCells : totalCells,
    minDxPc : minDxPc,
    dtYears : dtYears,
    totalSteps : totalSteps,
    runtimeSeconds : runtimeS,
    runtimeHours : runtimeS / 3600,
    runtimeDays : runtimeS / 86400,
    costGpuHours : cost,
    costSu : suCost
  };

  // Trigger Download
  const dataStr = "data:text/json;charset=utf-8," +
                  encodeURIComponent(JSON.stringify(exportData, null, 2));
  const downloadAnchorNode = document.createElement('a');
  downloadAnchorNode.setAttribute("href", dataStr);
  downloadAnchorNode.setAttribute("download", "quokka_runtime_estimation.json");
  document.body.appendChild(downloadAnchorNode); // required for firefox
  downloadAnchorNode.click();
  downloadAnchorNode.remove();
}
