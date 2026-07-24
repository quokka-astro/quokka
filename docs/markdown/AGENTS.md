# Documentation — Agent Guide

This directory (`docs/markdown/`) is the source for the Quokka documentation site, built with
[mdBook](https://rust-lang.github.io/mdBook/). Config lives in `docs/book.toml`; the built site goes
to `docs/site/` (CI builds and deploys it). Read this file before editing anything under
`docs/markdown/`.

Build/preview locally from the repo root:

```
mdbook build docs      # build into docs/site
mdbook serve docs      # live-preview at http://localhost:3000
```

## Adding a new page (READ THIS FIRST)

A Markdown file is **not** part of the book just because it exists on disk. Two indices must be
updated:

1. **`docs/markdown/SUMMARY.md` — required.** This is the mdBook table of contents. A page that is not
   listed in `SUMMARY.md` is **not included in the built book and will not render**, even if the file
   exists and is linked from other pages. This is the single most common mistake. `book.toml` sets
   `create-missing = false`, so mdBook will *not* auto-create ToC entries — every `SUMMARY.md` line
   must point to a real file, and every new file needs a `SUMMARY.md` line.
2. **The relevant in-page index** — for a test-problem page, add a bullet to
   `docs/markdown/tests/index.md` (the human-visible list of documented test problems). This is
   separate from, and additional to, the `SUMMARY.md` entry.

### Adding a new test-problem doc

Test-problem pages live in `docs/markdown/tests/`. When you add a new test problem (see the root
`AGENTS.md` / `CLAUDE.md` for the code side), also add its documentation:

1. Create `docs/markdown/tests/<ProblemName>.md`. A good page describes, in order: a short paragraph
   on the problem, the initial conditions, the boundary conditions, the analytic/reference solution
   (with the derivation), and the answer check (tolerances). See `tests/DTypeFront1D.md` and
   `tests/radshock.md` for the expected shape.
2. Add it to `docs/markdown/SUMMARY.md` under the `Test problems` list (nested bullet).
3. Add it to `docs/markdown/tests/index.md`.

Per-problem docs are optional (most problems do not have one); add a page when the problem has a
non-trivial analytic solution or answer check worth explaining.

## Math syntax

Math is rendered client-side by MathJax 3, loaded from `docs/javascripts/mathjax-init.js`. Note that
`book.toml` sets `mathjax-support = false` on purpose — the built-in mdBook MathJax is disabled and
replaced by this custom loader.

**In the Markdown source, use double backslashes** (a single backslash is consumed by the Markdown
parser before MathJax sees it):

- Inline math: `\\( ... \\)`  → e.g. `\\(x_i(t)\\)`
- Display math: `\\[ ... \\]`

These render to the `\( \)` / `\[ \]` delimiters that MathJax matches (configured in
`mathjax-init.js`).

**Legacy blocks are also supported** and are what most existing pages use. A shim in `mathjax-init.js`
converts them automatically:

```html
<script type="math/tex">           ...inline math...   </script>
<script type="math/tex; mode=display">   ...display math...   </script>
```

Prefer the native `\\( \\)` / `\\[ \\]` form for new inline/short expressions; the legacy `<script>`
blocks remain fine for multi-line display equations (e.g. `aligned` environments) and are used
throughout `tests/`.

**Subscript pitfall.** The Markdown parser can turn a `_` into emphasis (`<em>`) before MathJax runs,
which silently breaks subscripts. The legacy `<script>` shim repairs this automatically. With native
`\\( \\)` inline math, keep subscripts braced (`x_{\text{HII}}`, not a bare `x_ i`) or use a display
`\\[ \\]` block, which is not affected.

## Citations and diagrams

- **Citations:** pandoc-style `[@BibKey]`, resolved against `docs/markdown/references.bib` by the
  `mdbook-bib` preprocessor. A `[@key]` with no matching bib entry breaks the build — add the entry to
  `references.bib` first.
- **PlantUML:** fenced ```` ```plantuml ```` blocks render to inline SVG via the public PlantUML
  server (`preprocessor.plantuml` in `book.toml`).
- **Mermaid:** rendered client-side via `docs/javascripts/mermaid-init.js`.
