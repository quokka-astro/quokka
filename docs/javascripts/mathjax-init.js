(function() {
const mathJaxVersion = "3.2.2";
const mathJaxSrc = "https://cdn.jsdelivr.net/npm/mathjax@" + mathJaxVersion +
                   "/es5/tex-svg.js";

function convertLegacyMathScripts() {
  const scripts = document.querySelectorAll('script[type^="math/tex"]');

  for (const script of scripts) {
    const isDisplay = script.type.indexOf("mode=display") !== -1;
    const wrapper = document.createElement(isDisplay ? "div" : "span");
    const open = isDisplay ? "\\[" : "\\(";
    const close = isDisplay ? "\\]" : "\\)";
    // Markdown can turn TeX subscripts into literal <em>...</em> text before
    // MathJax sees inline script blocks. Convert those markers back to "_".
    const body = script.textContent.trim().replace(/<\/?em>/g, "_");

    wrapper.className = isDisplay ? "math math-display" : "math math-inline";
    wrapper.textContent = open + "\n" + body + "\n" + close;
    script.replaceWith(wrapper);
  }
}

function typesetMath() {
  convertLegacyMathScripts();

  if (!window.MathJax || typeof window.MathJax.typesetPromise !== "function") {
    return;
  }

  window.MathJax.typesetPromise().catch(
      (error) => { console.error("MathJax typesetting failed:", error); });
}

window.MathJax = {
  tex : {
    inlineMath : [ [ "\\(", "\\)" ] ],
    displayMath : [ [ "\\[", "\\]" ] ],
    processEscapes : true,
    processEnvironments : true,
  },
  options : {
    skipHtmlTags : [ "script", "noscript", "style", "textarea", "pre", "code" ],
  },
  svg : {
    fontCache : "global",
  },
  startup : {
    typeset : false,
  },
};

function ensureMathJax() {
  convertLegacyMathScripts();

  if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
    typesetMath();
    return;
  }

  const existing = document.querySelector('script[data-mathjax-loader="true"]');
  if (existing) {
    existing.addEventListener("load", typesetMath, {once : true});
    return;
  }

  const script = document.createElement("script");
  script.src = mathJaxSrc;
  script.async = true;
  script.dataset.mathjaxLoader = "true";
  script.addEventListener("load", () => {
    if (window.MathJax && window.MathJax.startup &&
        window.MathJax.startup.promise) {
      window.MathJax.startup.promise.then(typesetMath)
          .catch(
              (error) => { console.error("MathJax startup failed:", error); });
      return;
    }

    typesetMath();
  }, {once : true});
  document.head.appendChild(script);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", ensureMathJax, {once : true});
} else {
  ensureMathJax();
}
})();
