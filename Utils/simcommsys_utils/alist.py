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

from dataclasses import dataclass
from enum import Enum
import random


class AlistFormat(str, Enum):
    BINARY = "binary"
    NON_BINARY = "non-binary"
    SIMCOMMSYS = "simcommsys"


class ValuesMethod(str, Enum):
    ONES = "ones"
    RANDOM = "random"
    PROVIDED = "provided"


def chunks(xs, n: int):
    i = 0
    while i < len(xs):
        yield xs[i, i + n]
        i += n


@dataclass
class PchkMatrixAlist:
    """
    Dataclass representing a parity check matrix

    There are methods for easily reading such a dataclass from a file in alist or simcommsys format,
    and to dump into simcommsys format, or into another alist.
    """

    n: int  # length of a code word.
    m: int  # number of parity checks.
    max_col_weight: (
        int  # maximum number of parity checks that a particular symbol n participes in.
    )
    max_row_weight: (
        int  # maximum number of symbols that participate in a parity check m.
    )
    col_weights: list[int]  # Number of parity checks each symbol participates in.
    row_weights: list[int]  # Number of bits in each parity check.
    col_non_zero_pos: list[
        list[int]
    ]  # Non-zero positions for each column of the pchk matrix.
    row_non_zero_pos: list[
        list[int]
    ]  # Non-zero positions for each row of the pchk matrix.
    col_non_zero_values: list[
        list[int]
    ]  # Non-zero values for each column of the pchk matrix.
    row_non_zero_values: list[
        list[int]
    ]  # Non-zero values for each row of the pchk matrix.
    values_method: ValuesMethod
    random_seed: int = 0

    @classmethod
    def read(cls, input: str, format: AlistFormat) -> "PchkMatrixAlist":
        lines = input.split("\n")
        lines = [x.strip() for x in lines]
        # remove comment lines
        lines = [x for x in lines if not x.startswith("#")]

        match format:
            case AlistFormat.BINARY:
                n, m = [int(x) for x in lines[0].split(" ", 2)]
                max_col_weight, max_row_weight = [
                    int(x) for x in lines[1].split(" ", 2)
                ]

                col_weights = [int(x) for x in lines[2].split(" ")]
                row_weights = [int(x) for x in lines[3].split(" ")]

                return cls(
                    n=n,
                    m=m,
                    max_col_weight=max_col_weight,
                    max_row_weight=max_row_weight,
                    col_weights=col_weights,
                    row_weights=row_weights,
                    col_non_zero_pos=[
                        [int(x) for x in lines[i].split(" ")] for i in range(4, 4 + n)
                    ],
                    row_non_zero_pos=[
                        [int(x) for x in lines[i].split(" ")]
                        for i in range(4 + n, 4 + n + m)
                    ],
                    col_non_zero_values=[[1 for _ in range(w)] for w in col_weights],
                    row_non_zero_values=[[1 for _ in range(w)] for w in row_weights],
                    values_method=ValuesMethod.ONES,
                )

            case AlistFormat.NON_BINARY:
                n, m = [int(x) for x in lines[0].split(" ", 2)]
                max_col_weight, max_row_weight = [
                    int(x) for x in lines[1].split(" ", 2)
                ]

                col_weights = [int(x) for x in lines[2].split(" ")]
                row_weights = [int(x) for x in lines[3].split(" ")]

                col_non_zero_pos, col_non_zero_values = zip(
                    *[
                        zip(
                            *[
                                (int(pos), int(val))
                                for pos, val in chunks(lines[i].split(" "), 2)
                            ]
                        )
                        for i in range(4, 4 + n)
                    ]
                )

                row_non_zero_pos, row_non_zero_values = zip(
                    *[
                        zip(
                            *[
                                (int(pos), int(val))
                                for pos, val in chunks(lines[i].split(" "), 2)
                            ]
                        )
                        for i in range(4 + n, 4 + n + m)
                    ]
                )

                return cls(
                    n=n,
                    m=m,
                    max_col_weight=max_col_weight,
                    max_row_weight=max_row_weight,
                    col_weights=col_weights,
                    row_weights=row_weights,
                    col_non_zero_pos=col_non_zero_pos,
                    row_non_zero_pos=row_non_zero_pos,
                    col_non_zero_values=col_non_zero_values,
                    row_non_zero_values=row_non_zero_values,
                    values_method=ValuesMethod.PROVIDED,
                )

            case AlistFormat.SIMCOMMSYS:
                n = int(lines[0])
                m = int(lines[1])
                max_col_weight = int(lines[2])
                max_row_weight = int(lines[3])

                values_method = ValuesMethod(lines[4])

                # read random seed if this is present
                random_seed: int = 0
                lineno: int = 5
                if values_method == ValuesMethod.RANDOM:
                    random_seed = int(lines[lineno])
                    lineno += 1

                # skip size of col weights vector
                lineno += 1
                col_weights = [int(x) for x in lines[lineno].split(" ")]
                lineno += 1

                # skip size of row weights vector
                lineno += 1
                row_weights = [int(x) for x in lines[lineno].split(" ")]
                lineno += 1

                col_non_zero_pos = []
                if values_method != ValuesMethod.PROVIDED:
                    col_non_zero_values = [[1 for _ in range(w)] for w in col_weights]
                else:
                    col_non_zero_values = []

                for col in range(n):
                    # skip size of line
                    lineno += 1
                    if values_method != ValuesMethod.PROVIDED:
                        col_non_zero_pos.append(
                            [int(x) for x in lines[lineno].split(" ")]
                        )
                    else:
                        col_non_zero_pos_and_values = [
                            int(x) for x in lines[lineno].split(" ")
                        ]
                        col_non_zero_pos.append(
                            col_non_zero_pos_and_values[: col_weights[col]]
                        )
                        col_non_zero_values.append(
                            col_non_zero_pos_and_values[col_weights[col] :]
                        )
                    lineno += 1

                row_non_zero_pos = [[] for _ in range(m)]
                if values_method != ValuesMethod.PROVIDED:
                    row_non_zero_values = [[1 for _ in range(w)] for w in row_weights]
                else:
                    row_non_zero_values = [[] for _ in range(m)]

                for col in range(n):
                    for row, val in zip(
                        col_non_zero_pos[col], col_non_zero_values[col]
                    ):
                        row_non_zero_pos[row].append(col)
                        if values_method == ValuesMethod.PROVIDED:
                            row_non_zero_values[row].append(val)

                ### Validity checks
                for col in range(n):
                    assert (
                        len(col_non_zero_pos[col]) == col_weights[col]
                    ), f"Number of non-zero positions in column {col} does not match expected weight {col_weights[col]}"
                    assert (
                        len(col_non_zero_values[col]) == col_weights[col]
                    ), f"Number of non-zero values in column {col} does not match expected weight {col_weights[col]}"

                for row in range(m):
                    assert (
                        len(row_non_zero_pos[row]) == row_weights[row]
                    ), f"Number of non-zero positions in row {row} does not match expected weight {row_weights[row]}"
                    assert (
                        len(row_non_zero_values[row]) == row_weights[row]
                    ), f"Number of non-zero values in row {row} does not match expected weight {row_weights[row]}"

                return cls(
                    n=n,
                    m=m,
                    max_col_weight=max_col_weight,
                    max_row_weight=max_row_weight,
                    col_weights=col_weights,
                    row_weights=row_weights,
                    col_non_zero_pos=col_non_zero_pos,
                    row_non_zero_pos=row_non_zero_pos,
                    col_non_zero_values=col_non_zero_values,
                    row_non_zero_values=row_non_zero_values,
                    values_method=values_method,
                    random_seed=random_seed,
                )

            case _:
                raise RuntimeError(f"Unrecognized alist format {format}")

    def populate_values(
        self,
        gfsize: int = 2,
    ):
        assert (
            gfsize & (gfsize - 1) == 0
        ), f"Galois field size (gfsize) must be a power of two; got {gfsize} instead."

        self.values_method = ValuesMethod.PROVIDED

        # we need to generate new random values
        for col in range(self.n):
            for i, row in enumerate(self.col_non_zero_pos[col]):
                val = random.randint(1, gfsize - 1)
                self.col_non_zero_values[col][i] = val
                for j, col_ in enumerate(self.row_non_zero_pos[row]):
                    if col_ == col:
                        break
                else:
                    raise RuntimeError(
                        f"Could not find column {col} in adjacency list for expected neighbour row {row}."
                    )
                self.row_non_zero_values[row][j] = val

    def set_values_method(self, values_method: ValuesMethod):
        self.values_method = values_method

    def set_random_seed(self, random_seed: int):
        self.random_seed = random_seed

    def write(self, format: AlistFormat) -> str:
        match format:
            case AlistFormat.BINARY:
                col_non_zero_pos_str = "\n".join(
                    [" ".join(map(str, x)) for x in self.col_non_zero_pos]
                )
                row_non_zero_pos_str = "\n".join(
                    [" ".join(map(str, x)) for x in self.row_non_zero_pos]
                )

                return f"""{self.n} {self.m}
{self.max_col_weight} {self.max_row_weight}
{' '.join(map(str, self.col_weights))}
{' '.join(map(str, self.row_weights))}
{col_non_zero_pos_str}
{row_non_zero_pos_str}
"""

            case AlistFormat.NON_BINARY:
                col_non_zero_pos_and_values_str = ""
                for col in range(self.n):
                    col_non_zero_pos_and_values_str += " ".join(
                        [
                            f"{pos} {val}"
                            for pos, val in zip(
                                self.col_non_zero_pos[col],
                                self.col_non_zero_values[col],
                            )
                        ]
                    )

                row_non_zero_pos_and_values_str = ""
                for row in range(self.m):
                    row_non_zero_pos_and_values_str += " ".join(
                        [
                            f"{pos} {val}"
                            for pos, val in zip(
                                self.row_non_zero_pos[row],
                                self.row_non_zero_values[row],
                            )
                        ]
                    )

                return f"""{self.n} {self.m}
{self.max_col_weight} {self.max_row_weight}
{' '.join(map(str, self.col_weights))}
{' '.join(map(str, self.row_weights))}
{col_non_zero_pos_and_values_str}
{row_non_zero_pos_and_values_str}
"""

            case AlistFormat.SIMCOMMSYS:
                values_method_str = f"""# Non-zero values (ones|random|provided)
{self.values_method}
"""
                if self.values_method == ValuesMethod.RANDOM:
                    values_method_str += f"""# Random seed
{self.random_seed}"""

                non_zero_pos_str = ""
                for col in range(self.n):
                    non_zero_pos_str += f"""{len(self.col_non_zero_pos[col])}
{' '.join(map(str, self.col_non_zero_pos[col]))}"""

                    # add values at the end of each row if these are required.
                    if self.values_method == ValuesMethod.PROVIDED:
                        non_zero_pos_str += " " + " ".join(
                            map(str, self.col_non_zero_values[col])
                        )

                    non_zero_pos_str += "\n"

                return f"""# Length (n)
{self.n}
# Dimension (m)
{self.m}
# Max column weight
{self.max_col_weight}
# Max row weight
{self.max_row_weight}
{values_method_str}
# Column weight vector
{len(self.col_weights)}
{' '.join(map(str, self.col_weights))}
# Row weight vector
{len(self.row_weights)}
{' '.join(map(str, self.row_weights))}
# Non zero positions per col
{non_zero_pos_str}
"""

            case _:
                raise RuntimeError(f"Unrecognized alist format {format}")

    def draw_tanner_graph_tikz(self) -> str:
        s = """\\usetikzlibrary{calc}
\\begin{tikzpicture}[shorten >=1pt,node distance=2.5cm,auto]%,on grid"""

        s += "   % Symbol nodes\n"
        s += "   \\node[circle,draw] (n1) {};\n"
        for i in range(1, self.n):
            s += f"   \\node[circle,draw,right of=n{i}] (n{i+1}) {{}};\n"

        s += "\n"
        s += "   % Parity check nodes\n"
        n_mid = self.n // 2
        m_mid = self.m // 2
        s += f"   \\node[rectangle,draw,below of=n{n_mid+1}] (z{m_mid+1}) {{}};\n"
        for i in range(m_mid + 1, self.m):
            s += f"   \\node[rectangle,draw,right of=z{i}] (z{i+1}) {{}};\n"
        for i in range(m_mid - 1, -1, -1):
            s += f"   \\node[rectangle,draw,left of=z{i+2}] (z{i+1}) {{}};\n"

        s += "\n"
        s += "   % Connections between symbols and parity checks\n"
        for col, non_zero_pos in enumerate(self.col_non_zero_pos):
            for row in non_zero_pos:
                s += f"   \\path [-,draw] (n{col+1}) -- (z{row});\n"

        s += "\\end{tikzpicture}\n"
        return s
