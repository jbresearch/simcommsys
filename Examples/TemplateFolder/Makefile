# Copyright (c) 2025 Mark Mizzi
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

all: timers simulators

.PHONY: help
help:
	@echo "Available targets are:"
	@echo "	all         Makes all timer and simulator files"
	@echo "	timers	    Makes timer files"
	@echo "	simulators  Makes simulator files"
	@echo "	clean-{timers,simulators}	Cleans timer or simulator files"
	@echo "	clean       Cleans all generated Simcommsys files"
	@echo "	run-jobfile-<jobfile>   Run all simulations in the given jobfile."

.PHONY: simulators timers

# Create Simcommsys simulator files from System files.
# NOTE: Remove --simcommsys-tag flag to reproduce results from paper.
simulators:
	poetry run simcommsys-utils make-simulators \
		--resultscollector errors_hamming --input-mode random --no-analyze-decode-iters \
		--simcommsys-tag development \
		Systems Simulators

# Create Simcommsys timer files from System files.
# NOTE: Remove --simcommsys-tag flag to reproduce results from paper.
timers:
	poetry run simcommsys-utils make-timers \
		--simcommsys-tag development \
		Systems Timers

### Cleaning

.PHONY: clean-simulators clean-timers clean

clean: clean-timers clean-simulators

clean-timers:
	rm -rf Timers/*

clean-simulators:
	rm -rf Simulators/*

### Running jobs

# Generic run command
run-jobfile-%: Jobfiles/%.yaml
	poetry run simcommsys-utils run-jobs \
		--config-file $<
