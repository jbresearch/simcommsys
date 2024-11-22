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

import logging
import os
import subprocess
from glob import glob
from typing import Annotated, Any, Literal, cast, List
import re
import sys
from enum import Enum

import typer
from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader  # type:ignore

from .executors import (
    SimcommsysJob,
    SlurmSimcommsysExecutor,
    SimcommsysExecutor,
    LocalSimcommsysExecutor,
    MasterSlaveSimcommsysExecutor,
)
from .alist import PchkMatrix, PchkMatrixFormat, ValuesMethod

app = typer.Typer()


@app.command()
def copy_binaries(
    remote: Annotated[
        str,
        typer.Argument(
            help="Remote host to copy binaries to. Format: <username>@<host>"
        ),
    ],
    simcommsys_tag: Annotated[
        str, typer.Option(help="Build tag of Simcommsys binaries to copy.")
    ] = "development",
    arch: Annotated[
        str, typer.Option(help="Architecture of target machine.")
    ] = "x86_64",
    binary: Annotated[
        List[str] | None, typer.Option(help="List of binaries to copy to remote.")
    ] = None,
    simcommsys_type: Annotated[
        List[str] | None,
        typer.Option(help="List of Simcommsys types to copy to remote."),
    ] = None,
):
    """
    This is a utility command which can be used to copy Simcommsys binaries to a remote.
    """

    logging.info(f"Making directory {remote}:~/bin.{arch}...")
    subprocess.run(f"ssh {remote} 'mkdir -p bin.{arch}'", shell=True)

    for b in binary or []:
        for t in simcommsys_type or []:
            assert simcommsys_type in [
                "debug",
                "release",
                "profile",
            ], f"Unknown simcommsys type {t} specified. Known types are debug, release or profile."
            if os.path.isfile(f"~/bin.{arch}/{b}.{simcommsys_tag}.{t}"):
                logging.info(f"Copying {b}.{simcommsys_tag}.{t}...")
                subprocess.run(
                    f"scp ~/bin.{arch}/{b}.{simcommsys_tag}.{t} {remote}:~/bin.{arch}",
                    shell=True,
                )
            else:
                logging.warning(f"{b}.{simcommsys_tag}.{t} not found, skipping...")


class InputMode(Enum):
    """
    Specifies how a Simcommsys simulator should generate input data for the communication system.
    """

    ZERO = "zero"
    RANDOM = "random"
    USER = "user"

    @property
    def code(self) -> int:
        match self:
            case self.ZERO:
                return 0
            case self.RANDOM:
                return 1
            case self.USER:
                return 2
            case _:
                raise RuntimeError(
                    f"Unrecognized input mode for simulator specified: {self}"
                )


@app.command()
def make_simulators(
    input_dir: Annotated[
        str, typer.Argument(help="Input directory containing Simcommsys system files.")
    ],
    output_dir: Annotated[
        str,
        typer.Argument(
            help="Output directory where Simcommsys simulator files will be generated."
        ),
    ],
    resultscollector: Annotated[
        str, typer.Option(help="Results collector for the simulators.")
    ],
    input_mode: Annotated[
        InputMode,
        typer.Option(help="Specify input mode for the simulator."),
    ],
    analyze_decode_iters: Annotated[
        bool,
        typer.Option(
            help="Should results be collected for each seperate decode iteration? Ignored for stream simulators (effectively always set to true)."
        ),
    ] = True,
    stream_reset_n: Annotated[
        List[int] | None,
        typer.Option(
            help="List of stream lengths to work with for reset. Only applicable to commsys_stream. Default is [1]."
        ),
    ] = None,
    stream_term_n: Annotated[
        List[int] | None,
        typer.Option(
            help="List of stream lengths to work with for termination. Only applicable to commsys_stream. Default is [1]."
        ),
    ] = None,
    random_seed: Annotated[
        int,
        typer.Option(help="Random seed to be included in output if needed."),
    ] = 0,
):
    """
    Create Simcommsys simulator files from a set of Simcommsys system files in --input-dir.
    """

    assert os.path.isdir(
        input_dir
    ), f"Input directory given {input_dir} does not exist."
    assert os.path.isdir(
        output_dir
    ), f"Output directory given {output_dir} does not exist."

    input_mode_code = input_mode.code

    for sysfile in os.listdir(input_dir):
        commsys: str
        with open(os.path.join(input_dir, sysfile), "r") as fl:
            commsys = fl.read()

        if match := re.match(r"commsys_stream[^<]*<(.*),vector,(.*)>", commsys):

            # Simulators without parameters: open-ended streams
            with open(
                os.path.join(
                    output_dir, f"{resultscollector}-{input_mode.value}-{sysfile}"
                ),
                "w",
            ) as fl:

                fl.write(
                    f"""commsys_stream_simulator<{match.group(1)},{resultscollector},{match.group(2)}>
# Version
2
# Analyze all decode iterations?
1
# Streaming mode (0=open, 1=reset, 2=terminated)
0
# Version
2
# Input mode (0=zero, 1=random, 2=user[seq])
{input_mode_code}
"""
                )

                if input_mode == InputMode.USER:
                    fl.write(
                        """#: input symbols - count
1
#: input symbols - values
1
"""
                    )

                fl.write(
                    f"""# Communication system
{commsys}"""
                )

            # Simulators with parameters: streams with reset
            for reset_frames in [int(x) for x in stream_reset_n or [1]]:
                with open(
                    os.path.join(
                        output_dir,
                        f"{resultscollector}-{input_mode.value}-stream-reset-{reset_frames}-{sysfile}",
                    ),
                    "w",
                ) as fl:
                    fl.write(
                        f"""commsys_stream_simulator<{match.group(1)},{resultscollector},{match.group(2)}>
# Version
2
# Analyze all decode iterations?
1
# Streaming mode (0=open, 1=reset, 2=terminated)
1
# Number of frames to reset
{reset_frames}
# Version
2
# Input mode (0=zero, 1=random, 2=user[seq])
{input_mode_code}
"""
                    )

                    if input_mode == InputMode.USER:
                        fl.write(
                            """#: input symbols - count
1
#: input symbols - values
1
"""
                        )

                    fl.write(
                        f"""# Communication system
{commsys}"""
                    )

            # Simulators with parameters: streams with termination
            for term_frames in [int(x) for x in stream_term_n or [1]]:
                with open(
                    os.path.join(
                        output_dir,
                        f"{resultscollector}-{input_mode.value}-stream-term-{term_frames}-{sysfile}",
                    ),
                    "w",
                ) as fl:
                    fl.write(
                        f"""commsys_stream_simulator<{match.group(1)},{resultscollector},{match.group(2)}>
# Version
2
# Analyze all decode iterations?
1
# Streaming mode (0=open, 1=reset, 2=terminated)
2
# Number of frames to terminate
{term_frames}
# Version
2
# Input mode (0=zero, 1=random, 2=user[seq])
{input_mode_code}
"""
                    )

                    if input_mode == InputMode.USER:
                        fl.write(
                            """#: input symbols - count
1
#: input symbols - values
1
"""
                        )

                    fl.write(
                        f"""# Communication system
{commsys}"""
                    )

        elif match := re.match(r"commsys[^<]*<(.*),(.*)>", commsys):
            with open(
                os.path.join(
                    output_dir, f"{resultscollector}-{input_mode.value}-{sysfile}"
                ),
                "w",
            ) as fl:
                fl.write(
                    f"""commsys_simulator<{match.group(1)},{resultscollector}>
# Version
2
# Analyze all decode iterations?
{1 if analyze_decode_iters else 0}
# Input mode (0=zero, 1=random, 2=user[seq])
{input_mode_code}
"""
                )
                if input_mode == InputMode.USER:
                    fl.write(
                        """#: input symbols - count
1
#: input symbols - values
1
"""
                    )

                fl.write(
                    f"""# Communication system
{commsys}"""
                )
        else:
            logging.warning(
                f"Skipping {sysfile} as it does not appear to be a valid system file."
            )


@app.command()
def make_timers(
    input_dir: Annotated[
        str, typer.Argument(help="Input directory containing Simcommsys system files.")
    ],
    output_dir: Annotated[
        str,
        typer.Argument(
            help="Output directory where Simcommsys timer files will be generated."
        ),
    ],
):
    """
    Create Simcommsys timer files from a set of Simcommsys system files in --input-dir.
    """

    assert os.path.isdir(
        input_dir
    ), f"Input directory given {input_dir} does not exist."
    assert os.path.isdir(
        output_dir
    ), f"Output directory given {output_dir} does not exist."

    for sysfile in os.listdir(input_dir):
        commsys: str
        with open(os.path.join(input_dir, sysfile), "r") as fl:
            commsys = fl.read()

        if match := re.match(r"commsys[^<]*<([^,>]*)", commsys):
            with open(
                os.path.join(output_dir, sysfile),
                "w",
            ) as fl:
                fl.write(
                    f"""commsys_timer<{match.group(1)}>
# Version
2
# Analyze all decode iterations?
0
# Input mode (0=zero, 1=random, 2=user[seq])
1
# Communication system
{commsys}"""
                )
        else:
            logging.warning(
                f"Skipping {sysfile} as it does not appear to be a valid system file."
            )


class ExitChartSimulatorType(Enum):
    """
    Type of exit chart simulator that user wants to generate through command make-exit-chart-simulators.
    """

    PARALLEL_CODEC = "parallel/codec"
    SERIAL_CODEC = "serial/codec"
    SERIAL_MODEM = "serial/modem"

    @property
    def code(self) -> int:
        match self:
            case self.PARALLEL_CODEC:
                return 0
            case self.SERIAL_CODEC:
                return 1
            case self.SERIAL_MODEM:
                return 2
            case _:
                raise RuntimeError(f"Unrecognized EXIT chart type {self}.")


@app.command()
def make_exit_chart_simulators(
    input_dir: Annotated[
        str, typer.Argument(help="Input directory containing Simcommsys system files.")
    ],
    output_dir: Annotated[
        str,
        typer.Argument(
            help="Output directory where Simcommsys exit chart simulator files will be generated."
        ),
    ],
    exit_chart_type: Annotated[
        ExitChartSimulatorType,
        typer.Option(help="Type of exit chart produced by the simulator."),
    ],
    system_param: Annotated[
        float,
        typer.Option(help="Specify system parameter for the exit chart simulator."),
    ],
):
    """
    Create Simcommsys exit chart simulator files from a set of Simcommsys system files in --input-dir.
    """

    assert os.path.isdir(
        input_dir
    ), f"Input directory given {input_dir} does not exist."
    assert os.path.isdir(
        output_dir
    ), f"Output directory given {output_dir} does not exist."

    exit_chart_type_code = exit_chart_type.code

    for sysfile in os.listdir(input_dir):
        commsys: str
        with open(os.path.join(input_dir, sysfile), "r") as fl:
            commsys = fl.read()

        if match := re.match(r"commsys[^<]*<([^,>]*)", commsys):
            with open(
                os.path.join(
                    output_dir,
                    f"exit_computer-{exit_chart_type_code}-{system_param}-{sysfile}",
                ),
                "w",
            ) as fl:
                # use exit chart computer simulator
                fl.write(
                    f"""exit_computer<{match.group(1)}>
# Version
2
# EXIT chart type (0=parallel/codec, 1=serial/codec, 2=serial/modem)
{exit_chart_type_code}
# Compute binary LLR statistics?
0
# System parameter
{system_param}
# Communication system
{commsys}
"""
                )
        else:
            logging.warning(
                f"Skipping {sysfile} as it does not appear to be a valid system file."
            )


class RealType(Enum):
    """
    Type of real to be used with LDPC code in invocation of make-ldpc-systems.
    """

    DOUBLE = "double"
    FLOAT = "float"


class SPAType(Enum):
    """
    SPA implementation to be used with LDPC code in invocation of make-ldpc-systems.
    """

    GDL = "gdl"
    GDL_CUDA = "gdl_cuda"
    TRAD = "trad"


@app.command()
def make_ldpc_systems(
    codes_dir: Annotated[
        str,
        typer.Argument(
            help="Path to directory of Simcommsys TXT files with LDPC codes."
        ),
    ],
    templates_dir: Annotated[
        str,
        typer.Argument(
            help="Path to directory with system template which the generated system files will be based on."
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Argument(
            help="Path to directory where generated system files will be placed."
        ),
    ],
    real_type: Annotated[
        List[RealType],
        typer.Option(
            help="Floating point types to use in LDPC instantiations. An instantiation will be generated for each of the specified types."
        ),
    ],
    gf: Annotated[
        List[str],
        typer.Option(
            help="Galois fields to use in LDPC instantiations. An instantiation will be generated for each of the specified fields."
        ),
    ],
    spa_type: Annotated[
        List[SPAType],
        typer.Option(
            help="SPA types to use in LDPC instantiations. An instantian will be generated for each of the specified SPA types."
        ),
    ],
):
    """
    Creates LDPC system files from the templates specified by --templates-dir and the LDPC codes specified by --codes-dir

    One system file is created for each combination of template file, code file, real type (specified by --real-type), Galois field (specified by --gf) and SPA type (specified by --spa-type).

    The system files are generated in the output directory specified by --output-dir and have a filename following the scheme <template-filename>.<gf>.<ldpc-code-filename>.<real-type>.<spa-type>.txt

    The template file should contain all system components up until the codec.

    If there are components which depend on the Galois field, the Galois field can be specified as {gf} in the template; this will then be substituted for the actual field when instantiating the template.
    """

    assert os.path.isdir(
        templates_dir
    ), f"Templates directory given {templates_dir} does not exist."
    assert os.path.isdir(
        codes_dir
    ), f"Codes directory given {codes_dir} does not exist."

    assert os.path.isdir(
        output_dir
    ), f"Output directory given {output_dir} does not exist."

    for template_file in os.listdir(templates_dir):
        if not template_file.endswith(".txt"):
            logging.warning(
                f"{template_file} does not appear to be a template file (as it does not have a .txt suffix), skipping..."
            )
            continue

        commsys_template: str
        with open(os.path.join(templates_dir, template_file), "r") as fl:
            commsys_template = fl.read()

        for code_file in os.listdir(codes_dir):
            if not code_file.endswith(".txt") and not code_file.endswith(".alist"):
                logging.warning(
                    f"{code_file} does not appear to be an alist file (as it does not have a .txt/.alist suffix), skipping..."
                )
                continue

            code: str
            with open(os.path.join(codes_dir, code_file), "r") as fl:
                code = fl.read()

            for r in real_type:
                for s in spa_type:
                    for g in gf:
                        commsys = (
                            commsys_template.format(gf=g)
                            + f"""## Codec
ldpc<{g},{r.value}>
# Version
5
# SPA type (trad|gdl|gdl_cuda)
{s.value}
# Number of iterations
100
# Clipping method (zero=replace only zeros, clip=replace values below almostzero)
clip
# Value of almostzero
1e-38
# Reduce generator matrix to REF? (true|false)
0
{code}
"""
                        )

                        with open(
                            os.path.join(
                                output_dir,
                                os.path.basename(template_file).removesuffix(".txt")
                                + "."
                                + g
                                + "."
                                + code_file.removesuffix(".txt")
                                + "."
                                + r.value
                                + "."
                                + s.value
                                + ".txt",
                            ),
                            "w",
                        ) as fl:
                            fl.write(commsys)


@app.command()
def convert_alist(
    *,
    input: Annotated[
        str,
        typer.Argument(help="Input alist file."),
    ],
    output: Annotated[
        str | None,
        typer.Argument(
            help="Output alist file. If no file is specified, the output is printed to stdout."
        ),
    ] = None,
    from_format: Annotated[
        PchkMatrixFormat,
        typer.Option(help="Format of the input alist file."),
    ],
    to_format: Annotated[
        PchkMatrixFormat, typer.Option(help="Required format for the output.")
    ],
    values_method: Annotated[
        ValuesMethod,
        typer.Option(help="Which values method should be used in the output?"),
    ] = ValuesMethod.RANDOM,
    gfsize: Annotated[
        int,
        typer.Option(
            help="Field size to use when generating values if needed. This is ignored if input file already provides values."
        ),
    ] = 2,
    random_seed: Annotated[
        int,
        typer.Option(help="Random seed to be included in output if needed."),
    ] = 0,
):
    """
    This command can be used to convert one alist format to another.

    \b
    The currently supported formats are:
    - binary: This is the most commonly used format but can only accomodate binary LDPC codes. It was introduced and described by Mackay, e.g. in http://www.inference.org.uk/mackay/codes/alist.html
    - non-binary:
    - simcommsys: This is the format used internally by Simcommsys. It can accomodate binary/non-binary codes, and also specifies how non-binary values are generated if a binary code is to be used in a non-binary context.

    The command tries to accomodate the format conversions specified by the user as best as possible given the input file.

    \b
    If the input format is "binary", it will be read into memory with values_method set to "ones". If the user specifies --values-method as "provided" in this case, the command will set values_method to "provided" and generate non-zero values in the field specified by --gfsize.
    If the output format is then one which supports non-binary codes (such as "non-binary" or "simcommsys"), the generated values will show in this output; otherwise they are obviously discarded.
    If --values-method is specified as "random", the command will set values_method to "random" and set the random seed to that specified by --random-seed. This will then show in the output if the output format is "simcommsys".
    If --values-method is specified as "ones", nothing needs to be done, as the input already contains 1s as all non-zero values.

    \b
    If the input format is "non-binary", it will be read into memory with values_method set to "provided". If the user specifies --values-method as "provided" in this case, nothing will happen, as the input already provides non-zero values. The --gfsize argument will be ignored.
    If the user specifies --values-method as "ones", all non-zero values will be set to 1 in the output.
    If the user specifies --values-method as "random" the command will set values_method to "random" and set the random seed to that specified by --random-seed. This will then show in the output if the output format is "simcommsys".

    \b
    If the input format is "simcommsys" the behaviour is similar to the "binary" or "non-binary" case, depending on what the values_method specified in the input is.
    """

    assert os.path.isfile(input), f"Input file given {input} does not exist."
    assert not output or os.path.isdir(
        os.path.dirname(output)
    ), f"Directory for output file given {output} does not exist."

    alist: str
    with open(input) as fl:
        alist = fl.read()

    pchk = PchkMatrix.read(alist, from_format)
    if values_method != pchk.values_method:
        pchk.set_values_method(values_method)
        if values_method == ValuesMethod.RANDOM:
            # note that at this point we know that pchk does not
            # already have a random seed, as this would only be
            # the case if pchk.values_method == RANDOM to start
            # with.
            pchk.set_random_seed(random_seed)

        if values_method == ValuesMethod.PROVIDED:
            # note that at this point we know that pchk does not
            # already have non-zero values (other than 1), as this
            # would only be the case if pchk.values_method ==
            # PROVIDED to start with
            pchk.populate_values(gfsize)

        if values_method == ValuesMethod.ONES:
            pchk.populate_values(2)

    elif values_method == ValuesMethod.PROVIDED:
        # Warn against a common pitfall in usage
        logging.warning(
            f"The input file already has provided values; ignoring request to populate output alist with values from GF({gfsize})"
        )

    with open(output, "w") if output else sys.stdout as fl:
        fl.write(pchk.write(to_format))


@app.command()
def run_jobs(
    config_file: Annotated[
        str,
        typer.Option(
            help="Path to configuration file with details of Simcommsys jobs to run."
        ),
    ],
    group: Annotated[
        List[str] | None,
        typer.Option(
            help="Select groups of jobs to run from configuration file. All groups are run by default."
        ),
    ] = None,
):
    """
    This command can be used to run a batch of Simcommsys simulations.

    The user specifies a path to a configuration file (in YAML format) using --config-file. This file specifies the simulations to be run and must be in the following format:

    \b
    # specify executor (how Simcommsys simulations are run) and its details
    executor:
        type: slurm | local | masterslave
        # additional dynamic params that are specific to an executor type.
        [key: value]*
    # Specify simulations to be run.
    jobs:
        [
        <group-name>:
            # glob that specifies path to simulator files for all simulations in this group.
            # Paths must be specified relative to the configuration file.
            glob: <glob-path>
            # Output directory where simulation results will be stored.
            output_dir: <output-dir>
            # Simcommsys parameters for each simulation in this group
            start: <start>
            stop: <stop>
            confidence: <confidence>
            relative_error: <relative-error>
            floor_min: <floor-min>
            # Parameters that have to do with job execution
            simcommsys_tag: <simcommsys-tag>
            simcommsys_type: "release" | "debug" | "profile"
            # additional dynamic params that are specific to an executor type.
            [key: value]*
        ]*

    As shown in the above specification, Simcommsys simulations are grouped into named groups within the configuration file. The user can run just a select few of the groups specified in the configuration file using the --group option (by default all groups are run).

    For example if the configuration file contains the groups with names "cpu-eccperf", "gpu-eccperf", "cpu-timings" and "gpu-timings", the user can choose to run the simulations in the first two groups only by specifying --group cpu-eccperf --group gpu-eccperf.
    """

    assert os.path.isfile(
        config_file
    ), f"Specified configuration file {config_file} does not exist."

    config: dict[str, Any]
    with open(config_file, "r") as fl:
        config = load(fl, Loader=Loader)

    executor: SimcommsysExecutor
    match config["executor"].pop("type"):
        case "slurm":
            executor = SlurmSimcommsysExecutor(**config["executor"])
        case "local":
            executor = LocalSimcommsysExecutor(**config["executor"])
        case "masterslave":
            executor = MasterSlaveSimcommsysExecutor(**config["executor"])
        case _:
            raise RuntimeError("Unrecognized executor type.")

    jobs_by_group: dict[str, list[SimcommsysJob]] = {}
    for groupname, jobsspec in config["jobs"].items():
        output_dir = jobsspec.pop("output_dir")
        start = jobsspec.pop("start")
        stop = jobsspec.pop("stop")
        step = jobsspec.pop("step", None)
        mul = jobsspec.pop("mul", None)
        confidence = jobsspec.pop("confidence")
        relative_error = jobsspec.pop("relative_error")
        floor_min = jobsspec.pop("floor_min")

        # Input and output files are considered relative to the directory of the config file
        config_dir = os.path.dirname(config_file)
        jobs_by_group[groupname] = [
            SimcommsysJob(
                name=os.path.basename(jobfile.removesuffix(".txt")),
                inputfile=os.path.join(config_dir, jobfile),
                outputfile=os.path.join(
                    config_dir,
                    output_dir,
                    os.path.basename(jobfile).removesuffix(".txt") + ".json",
                ),
                start=start,
                stop=stop,
                step=step,
                mul=mul,
                confidence=confidence,
                relative_error=relative_error,
                floor_min=floor_min,
            )
            for jobfile in glob(jobsspec.pop("glob"), root_dir=config_dir)
        ]

    # by default all group names are selected
    selected_groups: set[str] = set(config["jobs"].keys())
    if group is not None:
        selected_groups = set(group)

    for groupname, jobsspec in config["jobs"].items():
        if groupname in selected_groups:
            executor.run(
                jobsspec.pop("simcommsys_tag"),
                cast(Literal["debug", "release"], jobsspec.pop("simcommsys_type")),
                jobs=jobs_by_group[groupname],
                **jobsspec,
            )
