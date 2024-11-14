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

import json
from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass
class Result:
    value: float
    errormargin: float


@dataclass
class Results:
    simcommsys_build: str
    confidence_level: int
    convergence_mode: str
    date: datetime
    system: str
    version: str
    by_param: dict[float, dict[str, float | int | Result]]

    @classmethod
    def loads(cls: type, s: str) -> "Results":
        d = json.loads(s)
        # transform any dicts into appropriate Results.
        by_param = {
            k1: {
                k2: (Result(**v2) if isinstance(v2, dict) else v2)
                for k2, v2 in v1.items()
            }
            for k1, v1 in d["results"].items()
        }
        return cls(
            simcommsys_build=d["Build"],
            confidence_level=float(d["Confidence Level"].removesuffix("%")) / 100,
            convergence_mode=d["Convergence Mode"],
            date=datetime.strptime(d["Date"], "%d %b %Y, %H:%M:%S"),
            system=d["System"],
            version=d["Version"],
            by_param=by_param,
        )

    def dumps(self) -> str:
        return json.dumps(
            {
                k1: {
                    k2: (asdict(r) if isinstance(r, Result) else r)
                    for k2, r in v1.items()
                }
                for k1, v1 in self.by_param.items()
            }
        )

    @classmethod
    def load(cls: type, f) -> "Results":
        return Results.loads(f.read())

    def dump(self, f):
        return f.write(self.dumps())
