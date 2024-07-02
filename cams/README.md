## Overview

This folder provides the necessary tools to benchmark the Copernicus Atmosphere Monitoring Service (CAMS) and plot data from it in a user-friendly animation.

## Features

- **Benchmarking Tools:** Perform a benchmark of CAMS and track the times for the operations of download, data processing and animation.
- **Data Visualization:** Visualize the concentration of PM10 (particles smaller than 10μm) in an animation for January 2023. This data is obtained from the CAMS European air quality reanalyses dataset.

## Usage

### Installing dependencies
Follow the instructions of the README.md at the high-level of this repository to install all dependencies.

#### Obtaining an API key
First, in order to access CAMS the user must obtain an API token by registering [on the platform](https://ads.atmosphere.copernicus.eu/) and navigating to their profile. A USER-ID (UID) and API Key will be presented. This information must be placed in environmental variables for the script to work.


#### Setting up environmental variables
* "CAMS_URL" which must be set to the URL of CAMS https://ads.atmosphere.copernicus.eu/api/v2
* "CAMS_API_KEY" which must contain the UID and API Key, seperated by a semicolon.
  * For example 12345:1a412b22-2b2d-12e2-abfc-17f1e7faa14a with 12345 being the UID.
  * **Note: This is just an example to demonstrate the format, you must obtain your own API key**

This can be done e.g. in python:
```python
import os

os.environ["CAMS_URL"] = https://ads.atmosphere.copernicus.eu/api/v2
os.environ["CAMS_API_KEY"] = 12345:1a412b22-2b2d-12e2-abfc-17f1e7faa14a
```

Or by using conda:

```bash
conda env config vars set CAMS_URL=https://ads.atmosphere.copernicus.eu/api/v2
conda env config vars set CAMS_API_KEY=12345:1a412b22-2b2d-12e2-abfc-17f1e7faa14a
```

<br />
<br />
<br />

Make sure to check that the environmental variables were correctly populated. 

In python:
```python
import os

print(os.environ["CAMS_URL"])
print(os.environ["CAMS_API_KEY"])
```

Or by using conda:

```bash
conda env config vars list | grep CAMS
```
### Running the benchmark
To run the benchmark, do the following

```Bash
python benchmark_cams.py
```

Results will be placed in the results folder. The script will autonomously download the data and clean the files after the completion of the benchmark. The data will be temporarily stored in the data folder
