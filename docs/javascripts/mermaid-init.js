(function() {
const mermaidVersion = "10.6.1";
const mermaidSrc = "https://cdn.jsdelivr.net/npm/mermaid@" + mermaidVersion +
                   "/dist/mermaid.min.js";

function convertBlocks() {
  const blocks = document.querySelectorAll("pre code.language-mermaid");

  for (const code of blocks) {
    const pre = code.parentElement;
    if (!pre || pre.dataset.mermaidConverted === "true") {
      continue;
    }

    const container = document.createElement("div");
    container.className = "mermaid";
    container.textContent = code.textContent;
    pre.dataset.mermaidConverted = "true";
    pre.replaceWith(container);
  }
}

// Detailed diagrams keep their controls outside the scrolling viewport.
function addDiagramControls() {
  for (const panel of document.querySelectorAll(".solver-flow")) {
    const svg = panel.querySelector("svg");
    if (!svg || panel.dataset.zoomReady) continue;
    const box = svg.viewBox.baseVal;
    if (!(box.width > 0 && box.height > 0)) continue;
    panel.dataset.zoomReady = "true";
    const width = box.width, height = box.height;
    const viewport = document.createElement("div");
    viewport.className = "solver-flow-viewport";
    viewport.tabIndex = 0;
    viewport.setAttribute("aria-label", "Scrollable flowchart");
    const diagram = panel.querySelector(".mermaid");
    viewport.appendChild(diagram);
    const toolbar = document.createElement("div");
    toolbar.className = "solver-flow-toolbar";
    toolbar.setAttribute("role", "group");
    toolbar.setAttribute("aria-label", "Flowchart zoom controls");
    const percentage = document.createElement("output");
    percentage.setAttribute("aria-live", "polite");
    let scale = 1, fitted = true;
    function setScale(value, fit = false) {
      const cx = (viewport.scrollLeft + viewport.clientWidth / 2) / scale;
      const cy = (viewport.scrollTop + viewport.clientHeight / 2) / scale;
      scale = Math.max(0.02, Math.min(4, value));
      fitted = fit;
      svg.style.setProperty("width", `${width * scale}px`);
      svg.style.setProperty("height", `${height * scale}px`);
      percentage.textContent = `${Math.round(scale * 100)}%`;
      viewport.scrollLeft = fit ? 0 : cx * scale - viewport.clientWidth / 2;
      viewport.scrollTop = fit ? 0 : cy * scale - viewport.clientHeight / 2;
    }
    function fitDiagram() {
      setScale(Math.min((viewport.clientWidth - 16) / width,
                        (viewport.clientHeight - 16) / height, 1), true);
    }
    function button(label, action) {
      const control = document.createElement("button");
      control.type = "button";
      control.textContent = label;
      control.addEventListener("click", action);
      toolbar.appendChild(control);
    }
    button("Zoom out", () => setScale(scale / 1.4));
    button("Zoom in", () => setScale(scale * 1.4));
    button("Fit whole chart", fitDiagram);
    button("100%", () => setScale(1));
    toolbar.appendChild(percentage);
    panel.append(toolbar, viewport);
    // Refit only in fit mode; preserve a reader's chosen zoom on resize.
    new ResizeObserver(() => { if (fitted) fitDiagram(); }).observe(viewport);
    fitDiagram();
  }
}

function renderMermaid() {
  if (!window.mermaid) {
    return;
  }

  window.mermaid.initialize({
    startOnLoad : false,
    theme : "default",
  });

  if (typeof window.mermaid.run === "function") {
    window.mermaid.run({querySelector : ".mermaid"}).then(addDiagramControls);
  } else if (typeof window.mermaid.init === "function") {
    window.mermaid.init(undefined, document.querySelectorAll(".mermaid"));
  }
}

function ensureMermaid() {
  if (document.querySelectorAll("pre code.language-mermaid").length === 0) {
    return;
  }

  convertBlocks();

  if (window.mermaid) {
    renderMermaid();
    return;
  }

  const existing = document.querySelector('script[data-mermaid-loader="true"]');
  if (existing) {
    existing.addEventListener("load", renderMermaid, {once : true});
    return;
  }

  const script = document.createElement("script");
  script.src = mermaidSrc;
  script.dataset.mermaidLoader = "true";
  script.addEventListener("load", renderMermaid, {once : true});
  document.head.appendChild(script);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", ensureMermaid, {once : true});
} else {
  ensureMermaid();
}
})();
