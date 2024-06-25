## Overview

This folder provides the necessary tools to benchmark the Copernicus Atmosphere Monitoring Service (CAMS) and plot data from it in a user-friendly animation.

## Features

- **Benchmarking Tools:** Perform a benchmark of CAMS and track the times for the operations of download, data processing and animation.
- **Data Visualization:** Visualize the concentration of PM10 (particles smaller than 10μm) in an animation for January 2023. 

## Usage

### Setup

First, in order to access CAMS the user must obtain an API token by registering on the platform and navigating to their profile. A USER-ID (UID) and API Key will be presented. This must be placed in the following environmental variables:

* "CAMS_URL" which must be set to the URL of CAMS https://ads.atmosphere.copernicus.eu/api/v2
* "CAMS_API_KEY" which must contain the UID and API Key, seperated by a semicolon. For example 12345:1a412b22-2b2d-12e2-abfc-17f1e7faa14a with 12345 being the UID. **Note: This is just an example to demonstrate the format, you must obtain your own API key**

This can be done e.g. in python:
```python
import os

os.environ["CAMS_URL"] = https://ads.atmosphere.copernicus.eu/api/v2
os.environ["CAMS_API_KEY"] = 12345:1a412b22-2b2d-12e2-abfc-17f1e7faa14a
```

Or in a Linux terminal

```bash
conda env config vars set CAMS_URL=https://ads.atmosphere.copernicus.eu/api/v2
conda env config vars set CAMS_API_KEY=12345:1a412b22-2b2d-12e2-abfc-17f1e7faa14a
```

You can run the following command to make sure the environmental variables are set correctly

```bash
conda env config vars list | grep CAMS
```
To run the benchmark, do the following
```Bash
python benchmark_cams.py
```

Results will be placed in the results folder.