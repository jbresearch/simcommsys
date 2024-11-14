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
    @abc.abstractmethod
    def _get_exec_cmd(
        self,
        jobname: str,
        cmd: str,
        **jobdetails,
    ) -> str:
        """
        Returns command used to execute a single simcommsys simulation on the remote.
        """
        raise NotImplementedError

    def _get_simcommsys_cmd(
        self, tag: str, type: Literal["debug", "release"], job: SimcommsysJob
    ):
        """
        Returns simcommsys command for a single simulation.
        """
        assert job.step or job.mul, "Must specify step or mul in SimmcommsysJob object."
        step_or_mul_opt: str
        if job.step:
            step_or_mul_opt = f"--step={job.step}"
        else:
            step_or_mul_opt = f"--mul={job.mul}"

        return f"simcommsys.{tag}.{type} \
                    -i {job.inputfile} -o {job.outputfile} \
                    --start {job.start} --stop {job.stop} {step_or_mul_opt} \
                    --confidence {job.confidence} --relative-error {job.relative_error} \
                    --floor-min {job.floor_min} \
                    -e local -f json"

    @abc.abstractmethod
    def run(
        self,
        tag: str,
        type: Literal["debug", "release"],
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

    def _get_exec_cmd(  # type:ignore
        self,
        jobname: str,
        cmd: str,
        *,
        needs_gpu: bool,
        n_gpus: int,
        gpuarch: str,
        memlimit_gb: int,
        timeout_mins: int,
        nodename: str | None,
    ) -> str:
        sbatch_opts = f"--mail-user={self.email} \
                        --job-name={jobname} \
                        --partition={self.gpu_partition if needs_gpu else self.cpu_partition} \
                        --ntasks=1 \
                        --cpus-per-task=1 \
                        --mem-per-cpu={memlimit_gb}G \
                        --time={timeout_mins} \
                        --output={jobname}.out \
                        --error={jobname}.err \
                        --account={self.account} \
                        --mail-type=all"
        if needs_gpu:
            sbatch_opts += f" --gres=gpu:{gpuarch}:{n_gpus}"
        if nodename:
            sbatch_opts += f" --nodelist={nodename}"

        return f"sbatch {sbatch_opts} --wrap='{cmd}'"

    def run(  # type:ignore
        self,
        tag: str,
        type: Literal["debug", "release"],
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
            cmd = self._get_exec_cmd(
                job.name,
                self._get_simcommsys_cmd(tag, type, job),
                needs_gpu=needs_gpu,
                n_gpus=n_gpus,
                gpuarch=gpuarch,
                memlimit_gb=memlimit_gb,
                timeout_mins=timeout_mins,
                nodename=(nodelist[index] if nodelist else None),
            )
            logging.debug(cmd)
            subprocess.run(cmd, shell=True)

            if nodelist:
                index = (index + 1) % len(nodelist)


class LocalSimcommsysExecutor(SimcommsysExecutor):
    def _get_exec_cmd(self, jobname: str, cmd: str, memlimit_gb=100) -> str:  # type: ignore
        return cmd

    def run(  # type:ignore
        self,
        tag: str,
        type: Literal["debug", "release"],
        jobs: list[SimcommsysJob],
        *,
        memlimit_gb=100,
    ):
        for job in jobs:
            cmd = self._get_exec_cmd(
                job.name,
                self._get_simcommsys_cmd(tag, type, job),
                memlimit_gb=memlimit_gb,
            )
            logging.debug(cmd)
            subprocess.run(cmd, shell=True)
