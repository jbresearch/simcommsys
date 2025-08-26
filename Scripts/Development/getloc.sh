#!/bin/bash

# main process
echo "Analysis for all C/C++ code:"
find . -type f \( -name '*.cpp' -or -name '*.c' -or -name '*.cu' -or -name '*.h' \) -print0 |xargs -0 c_count |tail -1

for developer in briffa wesemeyer mizzi abela; do
    echo "Analysis for code written by ${developer}:"
    find . -type f \( -name '*.cpp' -or -name '*.c' -or -name '*.cu' -or -name '*.h' \) -print0 |xargs -0 grep -i ${developer} -l |xargs c_count |tail -1
done
