# mdBook migration checklist

This file records the completed MkDocs to mdBook migration and the single remaining deployment check.

## Completed

- [x] Add mdBook configuration at `docs/book.toml` with `src = "markdown"`
- [x] Mirror the documentation structure in `docs/markdown/SUMMARY.md`
- [x] Add mdBook-specific local build and serve helpers
- [x] Add summary coverage validation with `scripts/check_mdbook_summary.py`
- [x] Add a wrapper chapter for the standalone runtime calculator
- [x] Add an mdBook 404 source page
- [x] Convert MkDocs-style `!!!` admonitions to Markdown that renders in mdBook
- [x] Enable Mermaid support via `docs/javascripts/mermaid-init.js`
- [x] Enable bibliography support for `[@key]` citations with `mdbook-bib`
- [x] Convert dollar-delimited math to mdBook-compatible inline and display forms
- [x] Update CI and GitHub Pages workflows to build mdBook output
- [x] Switch the mdBook build output from `docs/site-mdbook` to `docs/site`
- [x] Remove obsolete MkDocs configuration, dependencies, and helper scripts

## Remaining

- [ ] Let GitHub Pages deploy the mdBook site and validate live routing plus 404 handling

## Working commands

Build the mdBook site locally:

```bash
./scripts/bash/docs_build_mdbook.sh
```

Serve the mdBook site locally:

```bash
./scripts/bash/docs_build_and_view_mdbook.sh
```

Validate that all Markdown sources are represented in `SUMMARY.md`:

```bash
python3 scripts/check_mdbook_summary.py
```

## Local verification

- [x] `python3 scripts/check_mdbook_summary.py`
- [x] `./scripts/bash/docs_build_mdbook.sh`
