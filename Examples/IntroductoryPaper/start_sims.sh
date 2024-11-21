#!/bin/bash

set -e

cd ../../Utils

# Run jobs using local simulation mode
# poetry run simcommsys-utils run-jobs --config-file \
#    ../Examples/IntroductoryPaper/Configurations/local-jobs.yaml

# Run jobs using master-slave simulation mode with local workers
poetry run simcommsys-utils run-jobs --config-file \
   ../Examples/IntroductoryPaper/Configurations/masterslave-jobs.yaml