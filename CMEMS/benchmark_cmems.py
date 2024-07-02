#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This Python script to benchmark CMEMS data access service.
"""
import os
import time

from loguru import logger
from tqdm import tqdm

from utils import (Cmems, GridInterpolator, ProductdVisualizer, load_config,
                   plot_benchmark, save_results)

if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = load_config(file_path=os.path.join(dir_path, "config.yaml"))
    # Dictionary to store benchmarking results
    output_folder = config["output_folder"]
    num_requests = config["num_requests"]

    out_dir = os.path.join(dir_path, output_folder, "cmems")
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

    # Repeat benchmarking for a specified number of requests
    for r in tqdm(range(num_requests), desc="Processing requests",
                  unit="request", ncols=100, colour="#3eedc4"):

        t0 = time.time()
        try:
            c = Cmems(config["cmems_request_bgc"])
            bgc_path = c.download_data()
            c = Cmems(config["cmems_request_glo_phy"])
            glo_path = c.download_data()
            t1 = time.time()
        except Exception as e:
            logger.error(f"Issue in the data access or download: {e}")
            request_issues += 1
            continue
        gi = GridInterpolator(dataset_input_path=glo_path,
                              dataset_target_path=bgc_path,
                              variable="thetao")
        ds_glo, ds_bgc = gi.interpolate()
        t2 = time.time()
        param = {
            "log_norm": False,
            "cmap": "coolwarm",
            "title": "Sea water potential temperature [°C]",
            "unit": "°C"
                }
        pv = ProductdVisualizer(param=param, ds=ds_glo.thetao)
        anim = pv.generate_animation()
        t3 = time.time()
        # Record benchmarking times
        benchmark["download_time"][r] = (t1-t0)
        benchmark["data_processing"][r] = (t2-t1)
        benchmark["animation"][r] = (t3-t2)
        benchmark["end_to_end"][r] = (t3-t0)
        benchmark["request_issues"][r] = request_issues

        os.remove(os.path.join(dir_path, glo_path))
        os.remove(os.path.join(dir_path, bgc_path))

    title = 'End to End CMEMS products regriding + \
        animation generation benchmark'
    plot_benchmark(benchmark_dict=benchmark,
                   out_dir=out_dir, title=title)

    filename = os.path.join(out_dir, "cmems_benchmark.json")
    save_results(data=benchmark, filename=filename)
    logger.info(f"Benchmark completed. Results saved to {out_dir}",
                "benchmarks.json")
