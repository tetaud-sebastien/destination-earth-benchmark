import glob
import os

from loguru import logger
from tqdm import tqdm

from utils import (CdsERA5, WindSpeedVisualizer, load_config, plot_benchmark,
                   save_results)

if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = load_config(file_path=os.path.join(dir_path, "config.yaml"))

    # Dictionary to store benchmarking results
    output_folder = config["output_folder"]
    num_requests = config["num_requests"]

    out_dir = os.path.join(dir_path, output_folder, "cds")
    grib_path = os.path.join(out_dir, "tmp")

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

    query = config["cds_request"]
    cds = CdsERA5()
    # Repeat benchmarking for a specified number of requests
    for r in tqdm(range(num_requests),
                  desc="Processing requests",
                  unit="request", ncols=100,
                  colour="#3eedc4"):

        try:
            cds.get_data(query=query)
            cds.download(filename=grib_path)

        except Exception as e:
            logger.error(f"Issue in the data access or download: {e}")
            request_issues += 1
            continue
        wind_speed, ds = cds.process()
        wind_anim = WindSpeedVisualizer.generate_animation(wind_speed)

        # Record benchmarking times
        benchmark["download_time"][r] = cds.download.execution_time
        benchmark["data_processing"][r] = cds.process.execution_time
        benchmark["animation"][r] = WindSpeedVisualizer.generate_animation.execution_time
        benchmark["end_to_end"][r] = cds.get_data.execution_time + \
            cds.download.execution_time + \
            cds.process.execution_time + \
            WindSpeedVisualizer.generate_animation.execution_time

        benchmark["request_issues"][r] = request_issues

        for filename in glob.glob(f"{grib_path}*"):
            os.remove(filename)

    title = 'End to End ERA5 CDS wind speed animation generation benchmark'
    plot_benchmark(benchmark_dict=benchmark,
                   out_dir=out_dir, title=title)

    filename = os.path.join(out_dir, "cds_benchmark.json")
    save_results(data=benchmark, filename=filename)
    logger.info(f"Benchmark completed. Results saved to {out_dir}",
                "benchmarks.json")
