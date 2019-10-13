"""
Provide the definition of processing of SRTM DEM image.
"""
import gdal
import numpy as np
from image import Image
from filters.custom_filters import (DetectApplyFourier, BinaryClosing,
                                    GrovesCorrectionsIter)
from utils_dem import (resample_and_cut, unzip_resource)
from config_loader import Config


class SRTM(Image):  # pylint: disable=too-few-public-methods
    """
    Contain the elements and methods to perform the SRTM DEM processing.

    Attributes
    ----------
    srtm_zip : str
        Filepath to the zip file that contains the SRTM DEM image that
        include the area of interest
    srtm_tif : str
        Filepath where the SRTM DEM image in tif format is located
    srtm_interest : str
        Filepath with which the cropped of the area of interest made on SRTM
        will be stored
    fourier : Fourier
        Fourier component to perform fourier correction on SRTM
    groves : Groves
        Groves component to perform groves correction on SRTM

    Methods
    -------
    process
        Perform the processing related with SRTM image.
    """

    def __init__(self, area_of_interest):
        """
        Set attributes with values coming from parameter and configuration file
        values

        Parameters
        ----------
        area_of_interest : srt (pathfile)
            Path to shapefile of area of interest to process.
        """
        super().__init__(area_of_interest)
        self.srtm_zip = Config.srtm('SRTM_ZIP')
        self.srtm_tif = Config.srtm('SRTM_TIF')
        self.srtm_interest = Config.srtm('SRTM_AREA')
        self.fourier = Fourier(self.srtm_interest)
        self.groves = Groves(area_of_interest)

    def _prepare(self):
        """
        Some prepocessing needed to perform the SRTM processing. Unzip the zip
        file containing the SRTM DEM Image tile that include the area of
        interest. Resample and Cut SRTM using path to the shapefile and path
        and name with which the cropped area will be saved.
        """
        unzip_resource(self.srtm_zip)
        resample_and_cut(self.srtm_tif, self.aoi, self.srtm_interest)

    def process(self):
        """
        Call to private "prepare" method to perform the preprocessing necessary
        to perform the processing of SRTM image. Execute the fourier correction
        and use the output to correct the groves in SRTM image.

        Returns
        -------
        ndarray
            SRTM DEM image with Fourier and groves corrected cropped
            corresponding to area of interest.
        """
        self._prepare()
        fourier_corrected = self.fourier.process()
        return self.groves.process(fourier_corrected)


class Fourier(Image):  # pylint: disable=too-few-public-methods
    """
    Contain the elements and methods to perform the SRTM DEM Fourier
    correction.

    Attributes
    ----------
    srtm_interest : str
        Pathfile where the SRTM DEM image corresponding to area of interest can
        be found

    Methods
    -------
    process
        Perform the processing related with SRTM Fourier correction. Open the
        SRTM to correct as a ndarray object. Apply DetectApplyFourier Filter to
        execute the chain processing to perform the correction
    """

    def __init__(self, srtm_interest):
        """
        Set the area of interest to an instances attribute

        Parameters
        ----------
        srtm_interest : str
            Pathfile where the SRTM DEM image corresponding to area of interest
            can be found.
        """
        self.srtm_interest = srtm_interest

    def process(self):
        """
        Perform the processing related with SRTM Fourier correction. Open the
        SRTM to correct as a ndarray object. Apply DetectApplyFourier Filter to
        execute the chain processing to perform the correction

        Returns
        -------
        ndarray
            SRTM DEM image corresponding to area of interest with fourier
            correction applied
        """
        srtm_raw = gdal.Open(self.srtm_interest).ReadAsArray()
        return DetectApplyFourier().apply(srtm_raw)


class Groves(Image):  # pylint: disable=too-few-public-methods
    """
    Contain the elements and methods to perform the SRTM DEM groves correction.

    Attributes
    ----------
    groves_class_zip : str
        Filepath to the zip file that contains the Groves classification
        including the area of interest
    groves_class_tif : str
        Filepath where the Groves classification in tif format is located
    groves_interest
        Filepath with which the cropped of the area of interest made on Groves
        classification will be stored

    Methods
    -------
    process
        Perform the processing related with Groves correction. Call the prepare
        private method to get ready the resources to perform groves correction.
    """

    def __init__(self, area_of_interest):
        """
        Set attributes with values coming from parameter and configuration file
        values

        Parameters
        ----------
        area_of_interest : srt (pathfile)
            Path to shapefile of area of interest to process.
        """
        super().__init__(area_of_interest)
        self.groves_class_zip = Config.groves('GROVES_ZIP')
        self.groves_class_tif = Config.groves('GROVES_TIF')
        self.groves_interest = Config.groves('GROVES_AREA')

    def _prepare(self):
        """
        Some prepocessing needed to perform the Groves correction processing.
        Unzip the zip file containing the Groves classification including the
        area of interest. Resample and Cut Groves classification using path to
        the shapefile and path and name with which the cropped area will be
        saved. Open the cropped groves classification image and apply
        BinaryClosing filter.
        """
        unzip_resource(self.groves_class_zip)
        resample_and_cut(self.groves_class_tif, self.aoi, self.groves_interest)
        groves_class_raw = gdal.Open(self.groves_interest).ReadAsArray()
        return BinaryClosing(structure=np.ones((3, 3))).apply(groves_class_raw)

    def process(self, dem):
        """
        Perform the processing related with Groves correction. Call the prepare
        private method to get ready the resources to perform groves correction.
        Perform the Groves correction calling the GrovesCorrectionsIter Filter
        with groves classification and SRTM DEM image as parameters.

        Parameters
        ----------
        dem

        Returns
        -------
        ndarray
            SRTM DEM image corresponding to area of interest with fourier
            correction applied
        """
        srtm = dem
        groves_class = self._prepare()
        return GrovesCorrectionsIter(groves_class, iterations=3).apply(srtm)
