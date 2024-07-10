"""
Script to benchmark Copernicus Atmospheric Monitoring Service.
visualisation and animation of particles smaller than 10um.
"""
import os

from loguru import logger
from tqdm import tqdm

from utils import (
    CamsERA5,
    ParticleVisualizer,
    clean_directory,
    load_config,
    plot_benchmark,
    save_results)


def benchmark_cams():

    # Grab location of this file, change working directory and load config
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)
    config = load_config(file_path="config.yaml")

    # Dictionary to store benchmarking results

    output_folder = config["output_folder"]
    num_requests = config["num_requests"]

    out_dir = os.path.join(dir_path, output_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger.info("start benchmark")

    request_issues = 0

    benchmark = {
            "download_time": [None] * num_requests,
            "data_processing": [None] * num_requests,
            "animation": [None] * num_requests,
            "end_to_end": [None] * num_requests,
            "request_issues": [None] * num_requests,
            }

    query = config["cams_request"]
    cams = CamsERA5()

    # Repeat benchmarking for a specified number of requests
    for r in tqdm(
        range(num_requests),
        desc="Processing requests",
        unit="request",
        ncols=100,
        colour="#3eedc4"
    ):

        try:
            cams.get_data(query=query)
            cams.download(filename="CAMS")
        except Exception as e:
            logger.error(f"Issue in the data access or download: {e}")
            request_issues += 1
            continue
        ds = cams.process()
        _ = ParticleVisualizer.generate_animation(ds)

        # Record benchmarking times
        benchmark["download_time"][r] = cams.download.execution_time
        benchmark["data_processing"][r] = cams.process.execution_time
        benchmark["animation"][r] = \
            ParticleVisualizer.generate_animation.execution_time
        benchmark["end_to_end"][r] = \
            cams.get_data.execution_time + \
            cams.download.execution_time + \
            cams.process.execution_time + \
            ParticleVisualizer.generate_animation.execution_time

        benchmark["request_issues"][r] = request_issues

        clean_directory()

    title = 'End to End CAMS particle density animation generation benchmark'
    plot_benchmark(benchmark_dict=benchmark,
                   out_dir=out_dir,
                   title=title)

    filename = os.path.join(out_dir, "cams_benchmark.json")
    save_results(data=benchmark, filename=filename)
    logger.info(
        f"Benchmark completed. Results saved to {out_dir}",
        "benchmarks.json")


if __name__ == "__main__":

    benchmark_cams()
