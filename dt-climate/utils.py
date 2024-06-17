import time
import warnings

import earthkit.data
import earthkit.maps
import earthkit.regrid
import numpy as np

warnings.filterwarnings("ignore")


def polytope_benchmark(request_params, address, num_requests):
    """
    Run benchmarking for Polytope API.
    This function runs a benchmark for the Polytope API
    with given parameters and measuring the inference
    time for each request.

    Args:
        request_params (dict): A dictionary containing parameters
        for the API request.
        It should include keys like "class", "expver", "stream", as required by
        the Polytope API.
        address (str): The address of the Polytope server.
        num_requests (int): The number of requests to make for benchmarking.

    Returns:
        tuple: A tuple containing the average inference time and
        standard deviation.
        The first element is the average inference time in seconds,
        and the second
        element is the standard deviation in seconds.
    """
    # Initialize an empty list to store the inference times
    request_times = []

    for _ in range(num_requests):
        start_time = time.time()
        data = earthkit.data.from_source("polytope", "destination-earth",
                                         request_params,
                                         address=address,
                                         stream=False)
        end_time = time.time()
        request_times.append(end_time - start_time)

    # Compute average and standard deviation
    average_time = np.mean(request_times)
    std_deviation = np.std(request_times)

    return average_time, std_deviation, data

def get_dataset_size(ds):
    """
    Calculate the size of an xarray DataArray or Dataset in megabytes and gigabytes.

    Parameters:
    ds (xarray.DataArray or xarray.Dataset): The xarray object to calculate the size for.

    Returns:
    tuple: A tuple containing the size in megabytes and gigabytes.
    """
    size_in_bytes = ds.nbytes
    size_in_megabytes = size_in_bytes / (1024 * 1024)  # Convert bytes to megabytes
    size_in_gigabytes = size_in_bytes / (1024 * 1024 * 1024)  # Convert bytes to gigabytes
    
    return size_in_megabytes, size_in_gigabytes