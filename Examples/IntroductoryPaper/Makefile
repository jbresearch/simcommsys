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

simulators:
	cd ../../Utils; \
	poetry run simcommsys-utils make-simulators \
		--resultscollector errors_hamming --input-mode random \
		../Examples/IntroductoryPaper/Systems ../Examples/IntroductoryPaper/Simulators

clean:
	rm -rf Simulators/*