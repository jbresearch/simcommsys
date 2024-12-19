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
from typing import Any, Iterable
import logging
import re
from io import StringIO

import numpy as np


class PchkMatrixFormat(str, Enum):
    # Alist Binary format introduced by Mackay and Davey
    ALIST = "alist"
    # Format used by Simcommsys; this can accomodate both binary and non-binary codes.
    SIMCOMMSYS = "simcommsys"
    # A flat representation of the parity check matrix
    FLAT = "flat"


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
class PchkMatrix:
    cols: int
    rows: int

    row_non_zeros: np.ndarray[Any, np.int32]
    row_non_zero_pos: list[np.ndarray[Any, np.int32]]
    row_non_zero_val: list[np.ndarray[Any, np.int32]]

    col_non_zeros: np.ndarray[Any, np.int32]
    col_non_zero_pos: list[np.ndarray[Any, np.int32]]
    col_non_zero_val: list[np.ndarray[Any, np.int32]]

    random_seed: int | None = None

    @classmethod
    def __read_flat(
        cls, lines: Iterable[str], delimiter: str, transpose: bool
    ) -> "PchkMatrix":
        H = np.genfromtxt(lines, delimiter=delimiter, dtype=np.int32, unpack=transpose)

        rows, cols = H.shape

        return cls(
            cols=cols,
            rows=rows,
            row_non_zeros=np.sum(H != 0, axis=1, dtype=np.int32),
            row_non_zero_val=[H[r, H[r, :] != 0] for r in range(rows)],
            col_non_zeros=np.sum(H != 0, axis=0, dtype=np.int32),
            col_non_zero_val=[H[H[:, c] != 0, c] for c in range(cols)],
            row_non_zero_pos=[np.nonzero(H[r, :])[0] for r in range(rows)],
            col_non_zero_pos=[np.nonzero(H[:, c])[0] for c in range(cols)],
        )

    def __write_flat(self, delimiter: str, transpose: bool) -> str:
        H = np.zeros((self.rows, self.cols))
        for r in range(self.rows):
            H[r, self.row_non_zero_pos[r]] = self.row_non_zero_val[r]
        if transpose:
            H = H.T
        io = StringIO()
        np.savetxt(io, H, delimiter=delimiter)
        return io.read()

    @classmethod
    def __read_alist(cls, lines: Iterable[str]) -> "PchkMatrix":
        try:
            cols, rows = lines[0].split(" ", 1)
            try:
                cols, rows = int(cols), int(rows)
            except ValueError:
                raise RuntimeError(
                    f"Non-integer value given for rows or cols: {lines[0]}"
                )

            max_col_non_zeros, max_row_non_zeros = lines[1].split(" ", 1)
            try:
                max_col_non_zeros, max_row_non_zeros = int(max_col_non_zeros), int(
                    max_row_non_zeros
                )
            except ValueError:
                raise RuntimeError(
                    f"Non-integer value given for maximum row or col weight: {lines[1]}"
                )

            col_non_zeros = None
            try:
                col_non_zeros = np.array(list(map(int, lines[2].split(" "))))
            except ValueError:
                raise RuntimeError(
                    f"Non-integer value found in list of col non zeros: {lines[2]}"
                )

            row_non_zeros = None
            try:
                row_non_zeros = np.array(list(map(int, lines[3].split(" "))))
            except ValueError:
                raise RuntimeError(
                    f"Non-integer value found in list of row non zeros: {lines[3]}"
                )

            line_cntr = 4

            col_non_zero_pos = []
            col_non_zero_val: list[np.ndarray[Any, np.int32]]
            has_values = len(lines[line_cntr].split(" ")) == 2 * cols
            if has_values:
                col_non_zero_val = []
                for _ in range(cols):
                    pos, val = zip(*chunks(lines[line_cntr].split(" "), 2))
                    # do -1 as we use 0-based indexing
                    col_non_zero_pos.append(
                        np.array(list(map(lambda x: int(x) - 1, pos)), dtype=np.int32)
                    )
                    col_non_zero_val.append(np.array(list(map(int, val)), dtype=np.int32))

                    line_cntr += 1
            else:
                col_non_zero_val = [
                    np.array([1] * cw, dtype=np.int32) for cw in col_non_zeros
                ]
                for _ in range(cols):
                    pos = lines[line_cntr].split(" ")
                    col_non_zero_pos.append(np.array(list(map(int, pos)), dtype=np.int32))

                    line_cntr += 1

            row_non_zero_pos = []
            row_non_zero_val: list[np.ndarray[Any, np.int32]]
            has_values = len(lines[line_cntr].split(" ")) == 2 * rows
            if has_values:
                row_non_zero_val = []
                for _ in range(rows):
                    pos, val = zip(*chunks(lines[line_cntr].split(" "), 2))
                    # do -1 as we use 0-based indexing
                    row_non_zero_pos.append(
                        np.array(list(map(lambda x: int(x) - 1, pos)), dtype=np.int32)
                    )
                    row_non_zero_val.append(np.array(list(map(int, val)), dtype=np.int32))

                    line_cntr += 1
            else:
                row_non_zero_val = [
                    np.array([1] * rw, dtype=np.int32) for rw in row_non_zeros
                ]
                for _ in range(rows):
                    pos = lines[line_cntr].split(" ")
                    row_non_zero_pos.append(np.array(list(map(int, pos)), dtype=np.int32))

                    line_cntr += 1

            return cls(
                cols=cols,
                rows=rows,
                row_non_zeros=row_non_zeros,
                row_non_zero_val=row_non_zero_val,
                row_non_zero_pos=row_non_zero_pos,
                col_non_zeros=col_non_zeros,
                col_non_zero_val=col_non_zero_val,
                col_non_zero_pos=col_non_zero_pos,
            )

        except (IndexError, ValueError):
            raise RuntimeError("Invalid alist format.")

    def __write_alist(self) -> str:
        non_binary = max(*[np.max(r) for r in self.row_non_zero_val]) > 1

        col_non_zeros_str = " ".join(map(str, self.col_non_zeros))
        row_non_zeros_str = " ".join(map(str, self.row_non_zeros))

        col_non_zero_pos_and_values_str: str
        if non_binary:
            col_non_zero_pos_and_values_str = "\n".join(
                [
                    " ".join([f"{p+1} {v}" for p, v in zip(pos, val)])
                    for pos, val in zip(self.col_non_zero_pos, self.col_non_zero_val)
                ]
            )
        else:
            col_non_zero_pos_and_values_str = "\n".join(
                [
                    " ".join(map(lambda x: str(x + 1), pos))
                    for pos in self.col_non_zero_pos
                ]
            )

        row_non_zero_pos_and_values_str: str
        if non_binary:
            row_non_zero_pos_and_values_str = "\n".join(
                [
                    " ".join([f"{p+1} {v}" for p, v in zip(pos, val)])
                    for pos, val in zip(self.row_non_zero_pos, self.row_non_zero_val)
                ]
            )
        else:
            row_non_zero_pos_and_values_str = "\n".join(
                [
                    " ".join(map(lambda x: str(x + 1), pos))
                    for pos in self.row_non_zero_pos
                ]
            )

        return f"""{self.cols} {self.rows}
{np.max(self.col_non_zeros)} {np.max(self.row_non_zeros)}
{col_non_zeros_str}
{row_non_zeros_str}
{col_non_zero_pos_and_values_str}
{row_non_zero_pos_and_values_str}"""

    @staticmethod
    def __read_simcommsys_vector(lines: Iterable[str]) -> np.ndarray[Any, np.int32]:
        size = int(lines[0])
        vals = np.array(list(map(int, lines[1].split(" "))), dtype=np.int32)
        assert (
            size == vals.shape[0]
        ), f"Length of parsed array {vals.shape[0]} does not match expected size {size}"
        return vals

    @classmethod
    def __read_simcommsys(cls, lines: Iterable[str]) -> "PchkMatrix":
        # remove comment lines
        lines = map(lambda line: re.sub(r"#.*$", "", line).trim(), lines)
        lines = filter(lambda l: l == "", lines)
        lines = list(lines)

        cols = int(lines[0])
        rows = int(lines[1])

        values_method = ValuesMethod(lines[4])
        line_cntr = 5
        random_seed = None
        if values_method == ValuesMethod.RANDOM:
            random_seed = lines[line_cntr]
            line_cntr += 1

        col_non_zeros = cls.__read_simcommsys_vector(lines[line_cntr:])
        line_cntr += 2

        row_non_zeros = cls.__read_simcommsys_vector(lines[line_cntr:])
        line_cntr += 2

        col_non_zero_pos = []
        col_non_zero_val = []

        for c in range(cols):
            col_non_zero_pos.append(cls.__read_simcommsys_vector(lines[line_cntr:]))
            # we use 0-based indices not 1-based.
            col_non_zero_pos[-1] = np.array(
                list(map(lambda x: x - 1, col_non_zero_pos[-1]))
            )
            line_cntr += 2
            assert (
                col_non_zero_pos[-1].shape[0] == col_non_zeros[c]
            ), f"Size of position vector for column {c}={col_non_zero_pos[-1]}, expected {col_non_zeros[c]}"

            if values_method == ValuesMethod.PROVIDED:
                col_non_zero_val.append(
                    cls.__read_simcommsys_vector(lines[line_cntr:])
                )
                line_cntr += 2
                assert (
                    col_non_zero_val[-1].shape[0] == col_non_zero_pos[-1].shape[0]
                ), f"Sizes of position and value vectors for column {c} do not match {col_non_zero_val[-1].shape[0]} != {col_non_zero_pos[-1].shape[0]}"
            else:
                col_non_zero_val.append(np.ones((col_non_zeros[c],)))

        row_non_zero_pos = [[] for r in range(rows)]
        row_non_zero_val = [[] for r in range(rows)]

        for c in range(cols):
            for r, v in zip(col_non_zero_pos[c], col_non_zero_val[c]):
                row_non_zero_pos[r].append(c)
                row_non_zero_val[r].append(v)

        row_non_zero_pos = list(map(np.array, row_non_zero_pos))
        row_non_zero_val = list(map(np.array, row_non_zero_val))

        return cls(
            cols=cols,
            rows=rows,
            row_non_zeros=row_non_zeros,
            row_non_zero_val=row_non_zero_val,
            row_non_zero_pos=row_non_zero_pos,
            col_non_zeros=col_non_zeros,
            col_non_zero_val=col_non_zero_val,
            col_non_zero_pos=col_non_zero_pos,
            random_seed=random_seed,
        )

    @staticmethod
    def __write_simcommsys_vector(x: np.ndarray[Any, np.int32]) -> str:
        x_str = " ".join(map(str, x))
        return f"""{x.shape[0]}
{x_str}"""

    def __write_simcommsys(
        self, values_method: ValuesMethod, random_seed: int | None = None
    ) -> str:
        non_binary = max(*[np.max(r) for r in self.row_non_zero_val]) > 1
        # prohibit weird options that are probably mistakes
        assert (
            not non_binary or values_method != ValuesMethod.PROVIDED
        ), "Cannot specify that values are not provided when pchk matrix is non-binary"

        if non_binary and values_method != ValuesMethod.PROVIDED:
            logging.warning(
                "Values method specified was not provided but input provides values; these will be discarded."
            )

        values_method = f"""# Non-zero values (ones|random|provided)
{str(values_method)}"""

        if values_method == ValuesMethod.RANDOM:
            random_seed = random_seed or self.random_seed
            assert (
                random_seed is not None
            ), "Random seed must be specified as none was given in the input or as a command line parameter."
            values_method = f"""{values_method}
# Random seed
{random_seed}"""

        pos_and_values_str: str
        if values_method == ValuesMethod.PROVIDED:
            pos_and_values_str = "\n".join(
                [
                    f"""{self.__write_simcommsys_vector(np.array(list(map(lambda x: x+1, pos))))}
{self.__write_simcommsys_vector(val)}"""
                    for pos, val in zip(self.col_non_zero_pos, self.col_non_zero_val)
                ]
            )
        else:
            pos_and_values_str = "\n".join(
                map(
                    lambda pos: self.__write_simcommsys_vector(
                        np.array(list(map(lambda x: x + 1, pos)))
                    ),
                    self.col_non_zero_pos,
                )
            )

        return f"""# Length (n) 
{self.cols}
# Number of parity checks (m)
{self.rows}
# Maximum col weight
{np.max(self.col_non_zeros)}
# Maximum row weight
{np.max(self.row_non_zeros)}
{values_method}
# Column weights
{self.__write_simcommsys_vector(self.col_non_zeros)}
# Row weights
{self.__write_simcommsys_vector(self.row_non_zeros)}
# Per column positions (and perhaps values)
{pos_and_values_str}"""

    @classmethod
    def read(
        cls,
        inp: str,
        format: PchkMatrixFormat,
        delimiter: str = ",",
        transpose: bool = False,
    ) -> "PchkMatrix":
        match format:
            case PchkMatrixFormat.FLAT:
                return cls.__read_flat(
                    inp.split("\n"), delimiter=delimiter, transpose=transpose
                )
            case PchkMatrixFormat.ALIST:
                return cls.__read_alist(inp.split("\n"))
            case PchkMatrixFormat.SIMCOMMSYS:
                return cls.__read_simcommsys(inp.split("\n"))

    def write(
        self,
        format: PchkMatrixFormat,
        delimiter: str = ",",
        transpose: bool = False,
        values_method: ValuesMethod = ValuesMethod.RANDOM,
        random_seed: int | None = None,
    ) -> str:
        match format:
            case PchkMatrixFormat.FLAT:
                return self.__write_flat(delimiter=delimiter, transpose=transpose)
            case PchkMatrixFormat.ALIST:
                return self.__write_alist()
            case PchkMatrixFormat.SIMCOMMSYS:
                return self.__write_simcommsys(
                    values_method=values_method, random_seed=random_seed
                )
