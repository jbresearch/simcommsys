#!/bin/bash

set -e

cd ../../Scripts/SimcommsysUtils
poetry run simcommsys-utils run-jobs --config-file \
   ../../Examples/IntroductoryPaper/Configurations/local-jobs.yaml
