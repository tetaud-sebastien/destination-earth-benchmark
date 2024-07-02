#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This Python utils file contains functions for data loading, preprocessing,
visualization data from CMEMS.
"""
import json
import os
import re

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import copernicusmarine as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyinterp.backends.xarray
import xarray as xr
import yaml
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from IPython.display import HTML
from loguru import logger
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm


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


class ProductdVisualizer:

    def __init__(self, param: dict, ds):

        self.ds = ds
        self.log_norm = param["log_norm"]
        self.cmap = param["cmap"]
        self.title = param["title"]
        self.unit = param["unit"]

        if self.log_norm:
            self.norm = LogNorm(self.ds.min(), self.ds.max())
        else:
            self.norm = None

    def plot_product(self):
        """
        Plot wind speed data on a map.
        """
        lat_pattern = re.compile(r'lat(itude)?', re.IGNORECASE)
        lon_pattern = re.compile(r'lon(gitude)?', re.IGNORECASE)
        coord_names = self.ds[0].coords.keys()
        lat_name = find_coord_name(coord_names, lat_pattern)
        lon_name = find_coord_name(coord_names, lon_pattern)

        # Plotting
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)

        # Creating a heatmap

        heatmap = ax.pcolormesh(self.ds[lon_name], self.ds[lat_name],
                                self.ds, norm=self.norm,
                                cmap=self.cmap, transform=ccrs.PlateCarree())

        datetime_value = pd.to_datetime(self.ds.time.values)
        formatted_datetime = datetime_value.strftime('%Y-%m-%d %H:%M')
        # Adding title and axis labels directly to the ax object
        ax.set_title(f'{self.title}: {formatted_datetime}', fontsize=16)
        cax = fig.add_axes([ax.get_position().x1 + 0.02, ax.get_position().y0,
                            0.02, ax.get_position().height])

        cbar = plt.colorbar(heatmap, cax=cax, pad=1)
        cbar.set_label(f"[{self.unit}]")

        ax.add_feature(cartopy.feature.COASTLINE)
        ax.add_feature(cartopy.feature.LAND, zorder=10, facecolor='grey')
        ax.set_extent([self.ds[lon_name].min(), self.ds[lon_name].max(),
                       self.ds[lat_name].min(), self.ds[lat_name].max()],
                      crs=cartopy.crs.PlateCarree())
        gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), linewidth=0.5,
                          color='k', alpha=1,
                          linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_left = True
        gl.ylabels_right = True
        gl.xlines = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        plt.show()

    def generate_animation(self):
        """
        Generate an animation of wind speed data.
        """
        lat_pattern = re.compile(r'lat(itude)?', re.IGNORECASE)
        lon_pattern = re.compile(r'lon(gitude)?', re.IGNORECASE)
        coord_names = self.ds.coords.keys()
        lat_name = find_coord_name(coord_names, lat_pattern)
        lon_name = find_coord_name(coord_names, lon_pattern)
        # Detect latitude, longitude, and time coordinates
        lat_name = find_coord_name(coord_names, lat_pattern)
        lon_name = find_coord_name(coord_names, lon_pattern)
        # Plotting
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        heatmap = ax.pcolormesh(self.ds[lon_name], self.ds[lat_name],
                                self.ds.isel(time=0), norm=self.norm,
                                cmap=self.cmap, transform=ccrs.PlateCarree())

        datetime_value = pd.to_datetime(self.ds.time.values)
        formatted_datetime = datetime_value.strftime('%Y-%m-%d %H:%M')
        ax.set_title(f'{self.title}: {formatted_datetime}', fontsize=16)
        cax = fig.add_axes([ax.get_position().x1 + 0.02, ax.get_position().y0,
                            0.02, ax.get_position().height])
        # Initialize the plot elements
        mesh = ax.pcolormesh(self.ds[lon_name], self.ds[lat_name],
                                self.ds.isel(time=0), norm=self.norm,
                                cmap=self.cmap, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(heatmap, cax=cax, pad=1)
        cbar.set_label(f"[{self.unit}]")

        ax.add_feature(cartopy.feature.COASTLINE)
        ax.add_feature(cartopy.feature.LAND, zorder=10, facecolor='grey')
        ax.set_extent([self.ds[lon_name].min(), self.ds[lon_name].max(),
                       self.ds[lat_name].min(), self.ds[lat_name].max()],
                      crs=cartopy.crs.PlateCarree())
        gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), linewidth=0.5,
                          color='k', alpha=1,
                          linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_left = True
        gl.ylabels_right=True
        gl.xlines = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # Function to update the plot for each frame of the animation
        def update(frame):
            # Update the properties of the existing plot elements
            mesh.set_array(self.ds.isel(time=frame).values.flatten())
            datetime_value = pd.to_datetime(self.ds.time.values[frame])
            formatted_datetime = datetime_value.strftime('%Y-%m-%d %H:%M')
            ax.set_title(f'{self.title}: {formatted_datetime}', fontsize=16)

            return mesh,

        # Create the animation
        animation = FuncAnimation(fig, update, frames=len(self.ds.time),
                                  interval=200, blit=True)

        # Display the animation
        plt.close()  # Close initial plot to prevent duplicate display
        return HTML(animation.to_html5_video())


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
    ax = means.plot(kind='bar', stacked=True, figsize=(16, 8), yerr=errors, capsize=5)

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


class GridInterpolator:
    """
    A class for interpolating variables from an input dataset onto the grid of a target dataset.

    Attributes:
        ds_input (xarray.Dataset): The input dataset containing the data to be interpolated.
        ds_target (xarray.Dataset): The target dataset whose grid is used for interpolation.
        variable (str): The name of the variable in the input dataset to interpolate.
        interpolator (pyinterp.backends.xarray.Grid3D): The interpolator object initialized with the input dataset variable.
    """

    def __init__(self, dataset_input_path, dataset_target_path, variable):
        """
        Initializes the GridInterpolator class with paths to the input and target datasets and the variable to interpolate.

        Args:
            dataset_input_path (str): Path to the input dataset file.
            dataset_target_path (str): Path to the target dataset file.
            variable (str): The variable in the input dataset to interpolate.

        Raises:
            FileNotFoundError: If the specified dataset files are not found.
            KeyError: If the specified variable is not found in the input dataset.
        """
        try:
            # Load the datasets
            self.ds_input = xr.open_dataset(dataset_input_path)
            self.ds_target = xr.open_dataset(dataset_target_path)
        except FileNotFoundError as e:
            logger.error(f"Dataset file not found: {e}")
            raise

        try:
            # Select the first depth level for simplicity
            self.ds_input = self.ds_input.sel(depth=self.ds_input.depth.values[0])
            self.ds_target = self.ds_target.sel(depth=self.ds_target.depth.values[0])
            self.variable = variable
            # Initialize the interpolator using the input dataset's variable
            self.interpolator = pyinterp.backends.xarray.Grid3D(self.ds_input[self.variable])
        except KeyError as e:
            logger.error(f"Variable not found in dataset: {e}")
            raise

    def interpolate(self):
        """
        Interpolates the specified variable from the input dataset onto the grid of the target dataset.

        Uses trivariate interpolation on the longitude, latitude, and time dimensions.

        Returns:
            tuple:
                - xarray.Dataset: The new dataset with the interpolated variable.
                - xarray.Dataset: The original target dataset.

        Raises:
            ValueError: If the interpolation fails due to incompatible grid dimensions or other issues.
        """
        try:
            # Extract the grid coordinates from the target dataset
            lon = self.ds_target.longitude.values
            lat = self.ds_target.latitude.values
            time = self.ds_target.time.values

            # Create a meshgrid for interpolation
            mx, my, mz = np.meshgrid(lon, lat, time, indexing='ij')

            # Perform trivariate interpolation
            trivariate = self.interpolator.trivariate(
                dict(longitude=mx.ravel(), latitude=my.ravel(), time=mz.ravel())
            )

            # Reshape the result to match the target grid shape
            trivariate = trivariate.reshape(mx.shape).T

            # Create a new xarray dataset with the interpolated data
            new_ds = xr.Dataset(
                {
                    self.variable: (["time", "latitude", "longitude"], trivariate)
                },
                coords={
                    "time": time,
                    "latitude": lat,
                    "longitude": lon
                },
                attrs=self.ds_input.attrs
            )

            return new_ds, self.ds_target
        except ValueError as e:
            logger.error(f"Interpolation failed: {e}")
            raise


class Cmems:
    """
    A class to handle downloading data from the Copernicus Marine Environment Monitoring Service (CMEMS).

    Attributes:
        query (dict): A dictionary containing parameters for the dataset to be downloaded.
    """

    def __init__(self, query):
        """
        Initializes the Cmems class with a query dictionary.

        Args:
            query (dict): Dictionary containing parameters for the dataset query.
            Expected keys are:
            - dataset_id (str): ID of the dataset.
            - dataset_version (str): Version of the dataset.
            - variables (list): List of variables to be downloaded.
            - minimum_longitude (float): Minimum longitude of the area of interest.
            - maximum_longitude (float): Maximum longitude of the area of interest.
            - minimum_latitude (float): Minimum latitude of the area of interest.
            - maximum_latitude (float): Maximum latitude of the area of interest.
            - start_datetime (str): Start date and time for the data (ISO 8601 format).
            - end_datetime (str): End date and time for the data (ISO 8601 format).
            - minimum_depth (float): Minimum depth of the area of interest.
            - maximum_depth (float): Maximum depth of the area of interest.
            - disable_progress_bar (bool): Whether to disable the progress bar.
            - output_filename (str): Name of the output file.
            - output_directory (str): Directory where the output file will be saved.
            - force_download (bool): Whether to force the download if the file already exists.
        """
        self.query = query

    def download_data(self):
        """
        Downloads data based on the query parameters and saves it to the specified file.

        Uses the cm.subset function to download data from CMEMS according to the query parameters.

        Returns:
            str: The path to the downloaded file.

        Raises:
            Exception: If there is an error during the download process.
        """
        try:
            cm.subset(
                dataset_url=None,  # Assuming dataset_url is not needed, hence set to None
                dataset_id=self.query["dataset_id"],
                dataset_version=self.query["dataset_version"],
                variables=self.query["variables"],
                minimum_longitude=self.query["minimum_longitude"],
                maximum_longitude=self.query["maximum_longitude"],
                minimum_latitude=self.query["minimum_latitude"],
                maximum_latitude=self.query["maximum_latitude"],
                start_datetime=self.query["start_datetime"],
                end_datetime=self.query["end_datetime"],
                minimum_depth=self.query["minimum_depth"],
                maximum_depth=self.query["maximum_depth"],
                disable_progress_bar=self.query["disable_progress_bar"],
                output_filename=self.query["output_filename"],
                output_directory=self.query["output_directory"],
                force_download=self.query["force_download"]
            )
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise

        file_path = os.path.join(self.query["output_directory"], self.query["output_filename"])
        return file_path