## Overview

This folder provides the necessary tools to benchmark DestinE data access services, including polytope and cacheB, and visualize the data on an interactive map using Folium. The project aims to facilitate the analysis and comparison of different data access methods, as well as the presentation of climate data in a user-friendly manner.

## Features

- **Benchmarking Tools:** Perform benchmarks between different DestinE data access services. CacheB and Polytope only supported at the moment.
- **Data Visualization:** Visualize climate data on an interactive map with markers for different capitals.
- **Forecast Plotting:** Generate and display time series forecast for specific locations.

## Usage

### Benchmarking Data Access Services

```Bash
python main.py
```
**main.py** takes as input a yaml file as input:

```yaml
request_nb: 2
cacheb_url: https://cacheb.dcms.e2e.desp.space/destine-climate-dt/SSP3-7.0-IFS-NEMO-0001-standard-sfc-v0.zarr
capital_coordinates: {
    "Vienna": [48.2082, 16.3738],
    "Brussels": [50.8503, 4.3517],
}
output_folder: "result"
```

where:

- **request_nb**: Number of request for the same product to perform statistics
- **cacheb_url**: cacheB url to access the dataset
- **capital_coordinates**: dictionary type that contain the name of cities and its associated lat lon
- **output_folder**: folder name that will be created in the root project foler.

