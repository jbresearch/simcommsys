#!/bin/bash

# main process
echo "Source files with DEBUG > 1:"
find . -type f \( -name '*.cpp' -or -name '*.c' -or -name '*.cu' -or -name '*.h' \) -print0 |xargs -0 grep -l -E '# *define *DEBUG *[2-9]'
