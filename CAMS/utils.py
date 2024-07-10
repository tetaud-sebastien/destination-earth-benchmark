#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This Python utils file contains functions for data loading, preprocessing,
visualization data from CDS.
"""
import os
import cdsapi
import json
import xarray as xr
import numpy as np
import yaml
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import re
import datetime
import zipfile
import glob
import time


def benchmark(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time() - t1
        wrapper.execution_time = t2  # Store execution time as an attribute
        logger.info(f"{func.__name__} ran in {t2} seconds")
        return result
    return wrapper


def find_coord_name(coord_names, pattern):
    """
    Function to find coordinate names using regex

    """
    for name in coord_names:
        if pattern.search(name):
            return name
    return None


def load_config(file_path: str) -> dict:
    """
    Load YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Dictionary containing configuration information.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def clean_directory():
    """
    Function to clean all zip and NetCDF files from directory

    """
    # Cleanup by deleting zip and nc files
    for filename in glob.glob("*CAMS*.zip*"):
        os.remove(filename)
    for filename in glob.glob("*cams*.nc*"):
        os.remove(filename)


class ParticleVisualizer:
    @staticmethod
    def plot_pm10_concentration(ds):
        """
        Plot particle density data on a map.
        """
        lat_pattern = re.compile(r'lat(itude)?', re.IGNORECASE)
        lon_pattern = re.compile(r'lon(gitude)?', re.IGNORECASE)
        coord_names = ds.coords.keys()
        # Detect latitude, longitude, and time coordinates
        lat_name = find_coord_name(coord_names, lat_pattern)
        lon_name = find_coord_name(coord_names, lon_pattern)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        heatmap = ax.pcolormesh(ds[lon_name], ds[lat_name], ds,
                                cmap='Blues', transform=ccrs.PlateCarree())
        cbar = plt.colorbar(heatmap, ax=ax, orientation='horizontal', pad=0.05)
        cbar.set_label('Particles PM10 [µg m-3]')
        plt.title(
            f'PM10 on {np.datetime_as_string(ds.time.values, unit="D")}',
            fontsize=16)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    @benchmark
    def generate_animation(ds):
        """
        Generate an animation of particle data.
        """
        lat_pattern = re.compile(r'lat(itude)?', re.IGNORECASE)
        lon_pattern = re.compile(r'lon(gitude)?', re.IGNORECASE)
        coord_names = ds.coords.keys()
        lat_name = find_coord_name(coord_names, lat_pattern)
        lon_name = find_coord_name(coord_names, lon_pattern)
        # Detect latitude, longitude, and time coordinates
        lat_name = find_coord_name(coord_names, lat_pattern)
        lon_name = find_coord_name(coord_names, lon_pattern)
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        heatmap = ax.pcolormesh(ds[lon_name],
                                ds[lat_name],
                                ds.pm10.isel(time=0),
                                cmap='Blues',
                                transform=ccrs.PlateCarree())
        cbar = plt.colorbar(heatmap, ax=ax, orientation='horizontal', pad=0.05)
        cbar.set_label('Particles PM10 [µg m-3]')
        ax.set_title('PM10 animation')

        # Initialize the plot elements
        mesh = ax.pcolormesh(ds[lon_name],
                             ds[lat_name],
                             ds.pm10.isel(time=0),
                             cmap='Blues',
                             transform=ccrs.PlateCarree())

        # Function to update the plot for each frame of the animation
        def update(frame):
            # Update the properties of the existing plot elements
            mesh.set_array(ds.pm10.isel(time=frame).values.flatten())
            ax.set_title(f"""
                        PM10 on
                         {
                            np.datetime_as_string(
                            ds.pm10.time[frame].values,
                            unit="D")
                            }
                        """)

            return mesh,

        # Create the animation
        animation = FuncAnimation(
            fig,
            update,
            frames=len(ds.time),
            interval=200,
            blit=True)

        # Display the animation
        plt.close()  # Close initial plot to prevent duplicate display

        return HTML(animation.to_html5_video())


class CamsERA5:
    def __init__(self):
        """
        """
        try:
            self.client = cdsapi.Client(
                url=os.environ["CAMS_URL"],
                key=os.environ["CAMS_API_KEY"])
            logger.info("Successfully logged on to Atmosphere Data Store")
        except Exception:
            logger.error("Could not log on to Atmosphere Data Store")

    @benchmark
    def get_data(self, query):
        """
        """
        name = query["name"]
        request = query["request"]
        self.format = query["request"]["format"]
        self.result = self.client.retrieve(name, request)
        return self.result

    @benchmark
    def download(self, filename):
        """
        """
        date = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        self.filename = f"{filename}-{date}.nc"
        self.filename_zip = f"{filename}-{date}.{self.format}"

        self.result.download(self.filename_zip)

    @benchmark
    def process(self):
        with zipfile.ZipFile(self.filename_zip, 'r') as zip_ref:
            zip_ref.extractall("")

        for filename in glob.glob("*.nc"):
            self.filename = filename
            print("Extracted to file: CAMS\\" + str(filename))

        ds = xr.open_dataset(self.filename, engine="netcdf4")

        return ds


def plot_benchmark(benchmark_dict: dict, out_dir: str, title: str):
    """
    Plot benchmark results as a stacked bar chart with error bars.

    Parameters:
        benchmark_dict (dict): Dictionary containing benchmark results.
        out_dir (str): Output directory to save the plot.

    Returns:
        None
    """
    # Convert benchmark dictionary to DataFrame
    df = pd.DataFrame(benchmark_dict)
    df = df.drop(columns=['request_issues'])

    # Calculate average and standard deviation
    means = df.mean()
    errors = df.std()
    # Plotting the stacked bar chart
    ax = means.plot(
        kind='bar',
        stacked=True,
        figsize=(16, 8),
        yerr=errors,
        capsize=5)

    # Set labels and title
    ax.set_ylabel('Time [s]')
    ax.set_xlabel('Benchmark steps')
    ax.set_title(title)

    # Save the figure
    filename = os.path.join(out_dir, "benchmark_barplot.svg")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def save_results(data: dict, filename: str):
    """
    Save a dictionary to a JSON file.

    Parameters:
        data (dict): Dictionary to be saved.
        filename (str): Name of the JSON file to save.

    Returns:
        None
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)
