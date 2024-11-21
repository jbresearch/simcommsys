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

import abc
import subprocess
from dataclasses import dataclass
from typing import Literal
import logging
from multiprocessing import cpu_count


@dataclass
class SimcommsysJob:
    name: str
    inputfile: str
    outputfile: str
    start: float
    stop: float
    step: float | None
    mul: float | None
    confidence: float
    relative_error: float
    floor_min: float


class SimcommsysExecutor(abc.ABC):
    def _get_simcommsys_cmd(
        self,
        simcommsys_tag: str,
        simcommsys_type: Literal["debug", "release"],
        job: SimcommsysJob,
    ):
        """
        Returns simcommsys command for a single simulation.

        NOTE that the -e flag is not given and must be provided by a subclass
        """
        assert job.step or job.mul, "Must specify step or mul in SimmcommsysJob object."
        step_or_mul_opt: str
        if job.step:
            step_or_mul_opt = f"--step={job.step}"
        else:
            step_or_mul_opt = f"--mul={job.mul}"

        return f"simcommsys.{simcommsys_tag}.{simcommsys_type} \
                    -i {job.inputfile} -o {job.outputfile} \
                    --start {job.start} --stop {job.stop} {step_or_mul_opt} \
                    --confidence {job.confidence} --relative-error {job.relative_error} \
                    --floor-min {job.floor_min} \
                    -f json"

    @abc.abstractmethod
    def run(
        self,
        simcommsys_tag: str,
        simcommsys_type: Literal["debug", "release"],
        jobs: list[SimcommsysJob],
        **jobparams,
    ):
        """
        Runs several simcommsys simulations on a remote.
        """
        pass


class SlurmSimcommsysExecutor(SimcommsysExecutor):
    def __init__(
        self,
        *,
        email: str,
        gpu_partition: str,
        cpu_partition: str,
        account: str,
    ):
        # SLURM specific global options.
        self.email = email
        self.gpu_partition = gpu_partition
        self.cpu_partition = cpu_partition
        self.account = account

    def run(  # type:ignore
        self,
        simcommsys_tag: str,
        simcommsys_type: Literal["debug", "release"],
        jobs: list[SimcommsysJob],
        *,
        needs_gpu: bool,
        # NOTE: Defaults are provided so that we don't need to specify them in the case that needs_gpu = False.
        n_gpus: int = 1,
        gpuarch: str = "ampere",
        memlimit_gb: int,
        timeout_mins: int,
        nodelist: list[str] | str | None = None,
    ):
        nodelist = nodelist.split(",") if isinstance(nodelist, str) else nodelist

        index = 0
        for job in jobs:
            # build sbatch command for this job
            nodename = nodelist[index] if nodelist else None
            sbatch_opts = f"--mail-user={self.email} \
                        --job-name={job.name} \
                        --partition={self.gpu_partition if needs_gpu else self.cpu_partition} \
                        --ntasks=1 \
                        --cpus-per-task=1 \
                        --mem-per-cpu={memlimit_gb}G \
                        --time={timeout_mins} \
                        --output={job.name}.out \
                        --error={job.name}.err \
                        --account={self.account} \
                        --mail-type=all"
            if needs_gpu:
                sbatch_opts += f" --gres=gpu:{gpuarch}:{n_gpus}"
            if nodename:
                sbatch_opts += f" --nodelist={nodename}"
            simcommsys_cmd = self._get_simcommsys_cmd(
                simcommsys_tag, simcommsys_type, job
            )
            cmd = f"sbatch {sbatch_opts} --wrap='{simcommsys_cmd} -e local'"

            logging.debug(cmd)
            subprocess.run(cmd, shell=True)

            if nodelist:
                index = (index + 1) % len(nodelist)


class LocalSimcommsysExecutor(SimcommsysExecutor):
    def run(  # type:ignore
        self,
        simcommsys_tag: str,
        simcommsys_type: Literal["debug", "release"],
        jobs: list[SimcommsysJob],
        *,
        memlimit_gb=100,
    ):
        for job in jobs:
            # build command to submit to shell.
            simcommsys_cmd = self._get_simcommsys_cmd(
                simcommsys_tag, simcommsys_type, job
            )
            cmd = f"ulimit -v {memlimit_gb * 1024 * 1024} && {simcommsys_cmd} -e local"

            logging.debug(cmd)
            subprocess.run(cmd, shell=True)


class MasterSlaveSimcommsysExecutor(SimcommsysExecutor):
    def run(  # type:ignore
        self,
        simcommsys_tag: str,
        simcommsys_type: Literal["debug", "release"],
        jobs: list[SimcommsysJob],
        *,
        memlimit_gb=100,
        port: int | str = 3008,
        workers: int | str = cpu_count(),
    ):
        try:
            workers = workers if isinstance(workers, int) else int(workers)
        except ValueError:
            logging.error(
                f"Invalid value supplied for workers: {workers}, must be integer."
            )
            exit(-1)

        try:
            port = port if isinstance(port, int) else int(port)
        except ValueError:
            logging.error(f"Invalid value supplied for port: {port}, must be integer.")
            exit(-1)

        for job in jobs:
            # build command to submit to shell.
            simcommsys_cmd = self._get_simcommsys_cmd(
                simcommsys_tag, simcommsys_type, job
            )
            launch_slaves = f"""
# Wait for server to open its port
timeout 30 sh -c 'until netcat -z localhost {port}; do sleep 1; done'

# Launch the workers in the bg
for i in {{1..{workers}}}
do
    simcommsys.{simcommsys_tag}.{simcommsys_type} -e localhost:{port} 1>/dev/null 2>&1 &
done
"""
            # the command:
            # 1. sets a memory limit using ulimit -v.
            # 2. starts the simcommsys master in a screen session
            # 3. starts the simcommsys slaves in the background
            # 4. brings the simcommsys master to the foreground
            cmd = f"""set -e
ulimit -v {memlimit_gb * 1024 * 1024}
screen -d -m -S "{port}.{simcommsys_tag}.{simcommsys_type}" {simcommsys_cmd} -e :{port}
SERVER_PID=$!

{launch_slaves}

handler () {{
    echo 'Killing {job.name} server...'
    screen -XS "{port}.{simcommsys_tag}.{simcommsys_type}" quit
}}

trap handler INT

echo "Started simulation of {job.name} with {workers} workers."
echo "Press Ctrl+C to interrupt..."
while screen -list | grep -q "{port}.{simcommsys_tag}.{simcommsys_type}"
do
    sleep 1
done
"""

            logging.debug(cmd)
            subprocess.run(cmd, shell=True)
