import json
import time
import yaml
import xarray as xr
from loguru import logger
from tqdm import tqdm
from utils import get_dataset_size

def cacheb_benchmark(dataset_url, num_requests, output_file ):
    
    
    
    # Initialize benchmark result lists with None values
    benchmark_results = {
        "access_time": [None] * num_requests,
        "data_select_time": [None] * num_requests,
        "download_time": [None] * num_requests,
        "end_to_end_times": [None] * num_requests,
        "pixel_number": [None] * num_requests,
        "products_size": [None] * num_requests,
        "issue_element": [None] * num_requests,
    }

    request_issues = 0

    for r in tqdm(range(num_requests), desc="Processing requests", unit="request", ncols=100, colour="#3eedc4"):
        t0_cacheb = time.time()
        try:
            data = xr.open_dataset(
                dataset_url,
                engine="zarr",
                storage_options={"client_kwargs": {"trust_env": "true"}},
                chunks={}
            )
            t1_cacheb = time.time()
            
            data = data.sel(time="2023-12-01T01:00:00.000000000").t2m
            t2_cacheb = time.time()
            
            ds = data.load()
            t3_cacheb = time.time()
            
            benchmark_results["access_time"][r] = t1_cacheb - t0_cacheb
            benchmark_results["data_select_time"][r] = t2_cacheb - t1_cacheb
            benchmark_results["download_time"][r] = t3_cacheb - t2_cacheb
            benchmark_results["end_to_end_times"][r] = t3_cacheb - t0_cacheb
            benchmark_results["pixel_number"][r] = len(ds.longitude) * len(ds.latitude)
            size_in_megabytes, _ = get_dataset_size(ds)
            benchmark_results["products_size"][r] = size_in_megabytes
            benchmark_results["issue_element"][r] = "NAN"
            
        except Exception as err:
            request_issues += 1
            logger.error(f"Unexpected {err=}")
            benchmark_results["issue_element"][r] = str(err)

    benchmark_results["request_issues"] = request_issues

    # Convert and write JSON object to file
    with open(output_file, "w") as outfile:
        json.dump(benchmark_results, outfile)


if __name__ == "__main__":
    
    num_requests = 100
    dataset_url = "https://cacheb.dcms.e2e.desp.space/destine-climate-dt/SSP3-7.0-IFS-NEMO-0001-standard-sfc-v0.zarr"
    output_file = "cacheb_global_standard.json"

    cacheb_benchmark(dataset_url=dataset_url,
                        num_requests=num_requests,
                        output_file=output_file)
    
    dataset_url = "https://cacheb.dcms.e2e.desp.space/destine-climate-dt/SSP3-7.0-IFS-NEMO-0001-high-sfc-v0.zarr"
    output_file = "cacheb_global_high.json"
    cacheb_benchmark(dataset_url=dataset_url,
                        num_requests=num_requests,
                        output_file=output_file)
