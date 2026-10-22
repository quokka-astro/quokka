# Vendored copy of AMReX regression_testing

Upstream: https://github.com/AMReX-Codes/regression_testing
Commit: 65d942105b58ebee9008f180cbeede201d7a9e6f ("Merge pull request #155 from atmyers/benchmarkFile")

This directory was previously a git submodule. To update, replace its contents
with `git archive` of the desired upstream commit and update the hash above.

Only the files needed to run `regtest.py` are kept (plus README and LICENSE).
Removed from upstream: `.github/`, `docs/`, `deploy_docs_action.sh`, `.nojekyll`,
`reg_test_gc.py`, `requirements.txt`, `example-tests.ini`, `.gitignore`.
