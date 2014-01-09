#!/bin/bash

# main process
echo "Analysis for all C/C++ code:"
find . -type f \( -name '*.cpp' -or -name '*.c' -or -name '*.cu' -or -name '*.h' \) -print0 |xargs -0 c_count

echo "Analysis for code written by S. Wesemeyer:"
find . -type f \( -name '*.cpp' -or -name '*.c' -or -name '*.cu' -or -name '*.h' \) -print0 |xargs -0 grep -i wesemeyer -l |xargs c_count
