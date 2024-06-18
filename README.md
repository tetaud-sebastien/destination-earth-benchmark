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

```Bash
python authentification/cacheb-authentication.py -u username -p password >> ~/.netrc
python authentification/desp-authentication.py --user username --password password
```