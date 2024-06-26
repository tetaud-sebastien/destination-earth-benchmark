import copernicusmarine as cm
from loguru import logger
from utils import load_config
import os

class Cmems:

    def __init__(self, query):
        self.query = query

    def download_data(self):

        try:
            cm.subset(
            dataset_url = None,
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
            output_filename = self.query["output_filename"],
            output_directory = self.query["output_directory"],
            force_download = self.query["force_download"]
            )
        except Exception as e:
            logger.error(f"Error loading ERA5 data from Zarr store: {e}")
            raise


if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = load_config(file_path=os.path.join(dir_path,"config.yaml"))


    c = Cmems(config["cmems_request_bgc"])
    c.download_data()

    c = Cmems(config["cmems_request_glo_phy"])
    c.download_data()