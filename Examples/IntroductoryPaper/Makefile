# Copyright (c) 2024 Mark Mizzi
#
# This file is part of SimCommSys.
#
# SimCommSys is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SimCommSys is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.

all: simulators

.PHONY: simulators clean

run:
	poetry run python -m simcommsys_utils run-jobs --config-file \
   		Configurations/masterslave-jobs.yaml

run-local:
	poetry run python -m simcommsys_utils run-jobs --config-file \
   		Configurations/local-jobs.yaml

run-slurm:
	poetry run python -m simcommsys_utils run-jobs --config-file \
   		Configurations/slurm-jobs.yaml

simulators:
	poetry run python -m simcommsys_utils make-simulators \
		--resultscollector errors_hamming --input-mode random \
		Systems Simulators

clean:
	rm -rf Simulators/*