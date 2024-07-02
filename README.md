# Destination-Earth-Benchmark
## Prerequisites
1. Clone the repository:
    ```bash
    git clone git@github.com:tetaud-sebastien/destination-earth-benchmark.git
    ```
2. Install Python
    Download and install Python
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sh Miniconda3-latest-Linux-x86_64.sh
    ```
3. Install the required packages
    Create python environment:
    ```bash
    conda create --name env python==3.11
    ```
    Activate the environment

    ```bash
    conda activate env
    ```
    Install python package
    ```Bash
    pip install -r requirements.txt
    ```

    ```Bash
    conda install -c conda-forge ffmpeg
    ```

## Service authentification

1. Cache-B and Polytope for DT climate data

The user and password are DestinE password/username https://platform.destine.eu/home/

```Bash
python authentification/cacheb-authentication.py -u username -p password >> ~/.netrc
python authentification/desp-authentication.py --user username --password password
```

2. Copernicus marine

Get your user ID and API key from the CMEMS portal at the address
https://data.marine.copernicus.eu/register
```Bash
copernicusmarine login
```

3. ERA5

Get your user ID (UID) and API key from the CDS portal at the address https://cds.climate.copernicus.eu/user
and write it into the configuration file, so it looks like::

    $ cat ~/.cdsapirc
    url: https://cds.climate.copernicus.eu/api/v2
    key: <UID>:<API key>

Remember to agree to the Terms and Conditions of every dataset that you intend to download.


## Getting Help

Feel free to ask questions at the following email adress: sebastien.tetaud@esa.int or open a ticket.
