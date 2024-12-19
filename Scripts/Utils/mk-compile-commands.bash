#!/usr/bin/bash

make dry-run-build-libs-$1 | \
    grep -w 'gcc' | grep -w '\-c' | \
    jq -nR '[inputs|{directory:".", command:., file: match(" [^ ]+$").string[1:]}]' > .vscode/compile_commands.json

