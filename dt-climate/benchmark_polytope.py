import json
import time

import earthkit.data
import earthkit.regrid
import yaml
from loguru import logger
from tqdm import tqdm

from utils import get_dataset_size


def polytope_benchmark(config):


    request = config['request']
    num_requests = config['num_requests']
    output_file = config['output_file']
    address = config['address']
    grid = config['grid']

    # Initialize benchmark result lists with None values
    benchmark_results = {
        "download_time": [None] * num_requests,
        "transform_time": [None] * num_requests,
        "xarray_time": [None] * num_requests,
        "end_to_end_times": [None] * num_requests,
        "products_size": [None] * num_requests,
    }

    request_issues = 0
    logger.info("Starting benchmark with {} requests.", num_requests)

    for i in tqdm(range(num_requests), desc="Processing requests", unit="request", ncols=100, colour="#3eedc4"):
        t0_polytope = time.time()
        try:
            data = earthkit.data.from_source(
                "polytope", "destination-earth", request, address=address, stream=False
            )
            t1_polytope = time.time()

            out_grid = {"grid": [grid['lat'], grid['lon']]}
            data_latlon = earthkit.regrid.interpolate(
                data, out_grid=out_grid, method=grid['method']
            )
            t2_polytope = time.time()

            ds = data_latlon.to_xarray()
            t3_polytope = time.time()

            benchmark_results["download_time"][i] = t1_polytope - t0_polytope
            benchmark_results["transform_time"][i] = t2_polytope - t1_polytope
            benchmark_results["xarray_time"][i] = t3_polytope - t2_polytope
            benchmark_results["end_to_end_times"][i] = t3_polytope - t0_polytope

            size_in_megabytes, _ = get_dataset_size(ds)
            benchmark_results["products_size"][i] = size_in_megabytes

        except ValueError as e:
            logger.error(f"Issue in the data access or download: {e}")
            request_issues += 1

    benchmark_results["request_issues"] = request_issues

    with open(output_file, "w") as outfile:
        json.dump(benchmark_results, outfile)

    logger.info("Benchmark completed. Results saved to {}", output_file)


if __name__ == "__main__":
    logger.info("Loading configuration file.")
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    logger.info("Configuration loaded successfully.")
    polytope_benchmark(config)
