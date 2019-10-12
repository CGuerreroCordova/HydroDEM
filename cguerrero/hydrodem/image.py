"""
Provide the abstract class to subclass the Images classes.
"""

from abc import ABC, abstractmethod


class Image(ABC):
    """
    Abstract class.

    Attributes
    ----------
    aoi : str
        Path to the shapefile of area of interest to process

    Methods
    ------
    process
        Must be implemented for subclasses to process for different images.
    """
    def __init__(self, area_of_interest):
        """
        Set the area of interest to perform the processing.

        Parameters
        ----------
        area_of_interest : str
            Path to the shapefile of area of interest to process
        """
        self.aoi = area_of_interest

    @abstractmethod
    def process(self, *dem):
        raise NotImplementedError
