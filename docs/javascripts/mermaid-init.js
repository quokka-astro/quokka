(function() {
  const mermaidVersion = "10.6.1";
  const mermaidSrc = "https://cdn.jsdelivr.net/npm/mermaid@" + mermaidVersion + "/dist/mermaid.min.js";

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

  function renderMermaid() {
    if (!window.mermaid) {
      return;
    }

    window.mermaid.initialize({
      startOnLoad: false,
      theme: "default",
    });

    if (typeof window.mermaid.run === "function") {
      window.mermaid.run({ querySelector: ".mermaid" });
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
      existing.addEventListener("load", renderMermaid, { once: true });
      return;
    }

    const script = document.createElement("script");
    script.src = mermaidSrc;
    script.dataset.mermaidLoader = "true";
    script.addEventListener("load", renderMermaid, { once: true });
    document.head.appendChild(script);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", ensureMermaid, { once: true });
  } else {
    ensureMermaid();
  }
})();
