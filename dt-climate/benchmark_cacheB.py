import json
import time

import numpy as np
import xarray as xr
from loguru import logger

from utils import get_dataset_size

def cacheb_benchmark_roi(dataset_url, num_requests, num_roi, output_file ):
   
    
    benchmark_results = {}

    # Initiate a first request to get lat long
    DATSASET = xr.open_dataset(
        dataset_url,
        engine="zarr",
        storage_options={"client_kwargs": {"trust_env": "true"}},
        chunks={}
    )

    for roi in range(num_roi):
        select_times = []
        access_times = []
        download_times = []
        end_to_end_times = []
        products_size = []
        pixel_number = []
        request_issues = 0

        # Generate random min and max bounds for longitude and latitude
        lon_min, lon_max = sorted(np.random.uniform(low=DATSASET.longitude.values.min(), high=DATSASET.longitude.values.max(), size=2))
        lat_min, lat_max = sorted(np.random.uniform(low=DATSASET.latitude.values.min(), high=DATSASET.latitude.values.max(), size=2))
        
        logger.info(f"lat - long: {[lon_min, lon_max, lat_min, lat_max]}")

        for _ in range(num_requests):
            t0_cacheb = time.time()
            
            try:
                data = xr.open_dataset(
                    dataset_url,
                    engine="zarr",
                    storage_options={"client_kwargs": {"trust_env": "true"}},
                    chunks={}
                )
                t1_cacheb = time.time()
                
                # Data selection
                data = data.sel(time="2023-12-01T01:00:00.000000000").t2m
                data = data.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
                t2_cacheb = time.time()
                
                # Data download
                ds = data.load()
                t3_cacheb = time.time()
                
                access_times.append(t1_cacheb - t0_cacheb)
                select_times.append(t2_cacheb - t1_cacheb)
                download_times.append(t3_cacheb - t2_cacheb)
                end_to_end_times.append(t3_cacheb - t0_cacheb)
                size_in_megabytes, _ = get_dataset_size(ds)
                pixel_number.append(len(ds.longitude) * len(ds.latitude))
                products_size.append(size_in_megabytes)
                
            except Exception as err:
                request_issues += 1
                logger.error(f"Unexpected {err=}")
                continue

        logger.info(f"Product size: {size_in_megabytes}")
        benchmark_results[roi] = {
            "access_time": access_times,
            "data_select_time": select_times,
            "download_time": download_times,
            "end_to_end_times": end_to_end_times,
            "pixel_number": pixel_number,
            "products_size": products_size,
            "request_issues": request_issues
        }

    # Convert and write JSON object to file
    with open(output_file, "w") as outfile:
        json.dump(benchmark_results, outfile)
    
    logger.info("Benchmark completed. Results saved to {}", output_file)


if __name__ == "__main__":
    
    
    num_requests = 10
    num_roi = 200
    dataset_url = "https://cacheb.dcms.e2e.desp.space/destine-climate-dt/SSP3-7.0-IFS-NEMO-0001-standard-sfc-v0.zarr"
    output_file = "cacheb_roi_standard.json"

    cacheb_benchmark_roi(dataset_url=dataset_url,
                        num_requests=num_requests,
                        num_roi=num_roi,
                        output_file=output_file)
    
    dataset_url = "https://cacheb.dcms.e2e.desp.space/destine-climate-dt/SSP3-7.0-IFS-NEMO-0001-high-sfc-v0.zarr"
    output_file = "cacheb_roi_high.json"
    cacheb_benchmark_roi(dataset_url=dataset_url,
                        num_requests=num_requests,
                        num_roi=num_roi,
                        output_file=output_file)