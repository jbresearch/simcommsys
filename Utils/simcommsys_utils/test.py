#!/usr/bin/python3
import os
import glob
import json

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

SAMPLES = 10

SIMCOMMSYS_TAG = "development-omp-mpi-gmp-cuda75"
SIMCOMMSYS_TYPE = "release"

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIMULATORS = os.path.join(BASEDIR, "Simulators")
TIMERS = os.path.join(BASEDIR, "Timers")
FIGURES = os.path.join(BASEDIR, "Figures")


def run_quicksimulation(
    fieldsize: int,
    blocklen: int,
    spa_type: str,
    basedir: str,
    channel_param: float,
):
    path_rgx = os.path.join(
        basedir,
        f"*qsc-direct-straight-ldpc.gf{fieldsize}.{blocklen}.*.float.{spa_type}.txt",
    )
    paths = glob.glob(path_rgx)
    assert len(paths) == 1
    experiment_file = paths[0]

    with os.popen(
        f"quicksimulation.{SIMCOMMSYS_TAG}.{SIMCOMMSYS_TYPE} -i {experiment_file} -r {channel_param} --num-samples {SAMPLES} --output-format json 2>/dev/null"
    ) as resultsfile:
        return json.load(resultsfile)


def run_eccperf_simulator(
    fieldsize: int, blocklen: int, spa_type: str, channel_param: float
) -> tuple[int, int]:
    # Collect CPU results
    results = run_quicksimulation(field, blocklen, spa_type, SIMULATORS, channel_param)
    value, tolerance = (
        results["SER_99"]["Value"],
        results["SER_99"]["Tolerance"],
    )

    return value, tolerance


def run_timer(fieldsize: int, blocklen: int, spa_type: str) -> tuple[int, int]:
    results = run_quicksimulation(fieldsize, blocklen, spa_type, TIMERS, 1e-3)
    t_decode = 0
    err = 0
    for k, v in results.items():
        if k.startswith("t_decode"):
            t_decode += v["Value"]
            err += v["Tolerance"]

    return t_decode, err


if __name__ == "__main__":
    SMALL_FIELD = 2
    LARGE_FIELD = 512

    SMALL_BLOCKLEN = 1600
    LARGE_BLOCKLEN = 6400

    print("Timing performance")
    print("==================")

    # Plot Block length vs time.
    for field in [SMALL_FIELD, LARGE_FIELD]:
        BLOCKLENS = [400, 800, 1600, 3200, 6400]
        gdl_res = np.zeros((len(BLOCKLENS), 3))
        gdl_cuda_res = np.zeros((len(BLOCKLENS), 3))

        print(f"Blocklength vs time for GF({field})")
        for i, blocklen in enumerate(tqdm(BLOCKLENS)):
            gdl_res[i, :] = blocklen, *run_timer(field, blocklen, "gdl")
            gdl_cuda_res[i, :] = blocklen, *run_timer(field, blocklen, "gdl_cuda")

        plt.figure()
        plt.yscale("log")
        plt.xscale("log")
        plt.ylabel("Time")
        plt.xlabel("Block length")

        plt.errorbar(gdl_res[:, 0], gdl_res[:, 1], yerr=gdl_res[:, 2])
        plt.errorbar(gdl_cuda_res[:, 0], gdl_cuda_res[:, 1], yerr=gdl_cuda_res[:, 2])
        plt.legend(["gdl", "gdl_cuda"])

        plt.savefig(os.path.join(FIGURES, f"timings_gf{field}_blocklen_vs_time.png"))

    # Plot fieldsize vs time.
    for blocklen in [SMALL_BLOCKLEN, LARGE_BLOCKLEN]:
        FIELDS = [2, 8, 32, 128, 512]
        gdl_res = np.zeros((len(FIELDS), 3))
        gdl_cuda_res = np.zeros((len(FIELDS), 3))

        print(f"Field size vs time for N={blocklen}")
        for i, fieldsize in enumerate(tqdm(FIELDS)):
            gdl_res[i, :] = fieldsize, *run_timer(fieldsize, blocklen, "gdl")
            gdl_cuda_res[i, :] = fieldsize, *run_timer(fieldsize, blocklen, "gdl_cuda")

        plt.figure()
        plt.yscale("log")
        plt.xscale("log")
        plt.ylabel("Time")
        plt.xlabel("Field size")

        plt.errorbar(gdl_res[:, 0], gdl_res[:, 1], yerr=gdl_res[:, 2])
        plt.errorbar(gdl_cuda_res[:, 0], gdl_cuda_res[:, 1], yerr=gdl_cuda_res[:, 2])
        plt.legend(["gdl", "gdl_cuda"])

        plt.savefig(
            os.path.join(FIGURES, f"timings_bl{blocklen}_fieldsize_vs_time.png")
        )

    print("ECC Performance")
    print("===============")

    for field in [SMALL_FIELD, LARGE_FIELD]:
        for blocklen in [SMALL_BLOCKLEN, LARGE_BLOCKLEN]:
            print(f"Field size={field} Block length={blocklen}")

            gdl_res = np.zeros((8, 3))
            gdl_cuda_res = np.zeros((8, 3))

            for i in tqdm(range(1, gdl_res.shape[0] + 1)):
                channel_param = 0.5**i

                # Collect CPU results
                gdl_res[i - 1, :] = channel_param, *run_eccperf_simulator(
                    field, blocklen, "gdl", channel_param
                )

                # Collect GPU results
                gdl_cuda_res[i - 1, :] = channel_param, *run_eccperf_simulator(
                    field, blocklen, "gdl_cuda", channel_param
                )

            # Plot results
            plt.figure()

            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Channel Parameter")
            plt.ylabel("Symbol Error Rate")

            plt.errorbar(gdl_res[:, 0], gdl_res[:, 1], yerr=gdl_res[:, 2])
            plt.errorbar(
                gdl_cuda_res[:, 0], gdl_cuda_res[:, 1], yerr=gdl_cuda_res[:, 2]
            )
            plt.legend(["gdl", "gdl_cuda"])

            plt.savefig(os.path.join(FIGURES, f"eccperf_{field}_{blocklen}.png"))
