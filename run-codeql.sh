#!/bin/sh

codeql database create codeql-db --language=cpp --source-root=src --working-dir=build --command=ninja
codeql database analyze codeql-db cpp-lgtm.qls --output=cpp-analysis.sarif --format=sarifv2.1.0
