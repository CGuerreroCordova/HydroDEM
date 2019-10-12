"""
Provide the definition of processing of HSHEDS DEM image.
"""
import gdal
from image import Image
from filters.custom_filters import (LagoonsDetection, ClipLagoonsRivers,
                                    ProcessRivers)
from utils_dem import (resample_and_cut, unzip_resource, rasterize_rivers,
                       clip_lines_vector)
from config_loader import Config


class HSHEDS(Image):
    """
    Contain the elements and methods to perform the HSHEDS DEM processing.

    Attributes
    ----------
    hsheds_zip : str
        Filepath to the zip file that contains the HSHEDS DEM image that
        include the area of interest
    hsheds_adf :
        Filepath where the HSHEDS DEM image in adf format is located
    hsheds_tif : str
        Filepath where the HSHEDS DEM image in tif format is located
    hsheds_interest : str
        Filepath with which the cropped of the area of interest made on HSHEDS
        will be stored
    lagoons : Lagoons
        Fourier component to perform fourier correction on SRTM
    rivers : Rivers
        Groves component to perform groves correction on SRTM

    Methods
    -------
    process
        Perform the processing related with HSHEDS image.
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
        self.hsheds_zip = Config.hsheds('HSHEDS_ZIP')
        self.hsheds_adf = Config.hsheds('HSHEDS_ADF')
        self.hsheds_tif = Config.hsheds('HSHEDS_TIF')
        self.hsheds_interest = Config.hsheds('HSHEDS_AREA')
        self.lagoons = Lagoons(self.hsheds_interest)
        self.rivers = Rivers(area_of_interest)

    def _prepare(self):
        """
        Some prepocessing needed to perform the HSHEDS processing. Unzip the
        zip file containing the HSHEDS DEM Image tile in adf format that
        include the area of interest. Convert the HSHEDS file from adf to tif
        format Resample and Cut HSHEDS using path to the shapefile and path and
        name with which the cropped area will be saved.
        """
        unzip_resource(self.hsheds_zip)
        gdt_options = gdal.TranslateOptions(format='GTIFF')
        gdal.Translate(self.hsheds_tif, self.hsheds_adf, options=gdt_options)
        resample_and_cut(self.hsheds_tif, self.aoi, self.hsheds_interest)

    def process(self):
        """
        Call to private "prepare" method to perform the preprocessing necessary
        to perform the processing of SRTM image. Execute the lagoons detections
        process and call to the rivers process using result coming from
        processing lagoons before. Finally return this two results (lagoons and
        rivers)

        Returns
        -------
        tuple(ndarray, ndarray)
            Mask of lagoons and rivers detected.
        """
        self._prepare()
        lagoons = self.lagoons.process()
        rivers = self.rivers.process(lagoons)
        return lagoons, rivers


class Lagoons(Image):
    """
    Contain the elements and methods to perform the HSHEDS Lagoons detection

    Attributes
    ----------
    hsheds_interest : str
        Pathfile where the HSHEDS DEM image corresponding to area of interest
        can be found

    Methods
    -------
    process
        Perform the processing related with HSHEDS lagoons detection. Open the
        HSHEDS as a ndarray object, this is to perform the detection. Apply
        LagoonsDetection Filter to HSHEDS execute the chain processing to
        perform the lagoons detection
    """

    def __init__(self, area_of_interest):
        """
        Set the area of interest to an instances attribute

        Parameters
        ----------
        hsheds_interest : str
            Pathfile where the HSHEDS DEM image corresponding to area of
            interest can be found.
        """
        self.hsheds_interest = area_of_interest

    def process(self):
        """
        Perform the processing related with HSHEDS lagoons detection. Open the
        HSHEDS as a ndarray object, this is to perform the detection. Apply
        LagoonsDetection Filter to HSHEDS execute the chain processing to
        perform the lagoons detection

        Returns
        -------
        ndarray
            Binary Mask with lagoons detection made.
        """
        hydro_sheds = gdal.Open(self.hsheds_interest).ReadAsArray()
        lagoons = LagoonsDetection()
        lagoons.apply(hydro_sheds)
        return lagoons


class Rivers(Image):
    """
    Contain the elements and methods to perform the Rivers extraction and
    correction.

    Attributes
    ----------
    rivers_zip : str
        Filepath to the zip file that contains the complete vector rivers.
    rivers_full : str
        Filepath where the full vector rivers will be located
    rivers_tif : str
        Filepath where rasterized rivers in tif format will be located
    rivers_interest : str
        Filepath with which the cropped of the area of interest made on
        rasterized rivers will placed

    Methods
    -------
    process
        Perform the processing related with rivers detection and correction.
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
        self.rivers_zip = Config.rivers('RIVERS_ZIP')
        self.rivers_full = Config.rivers('RIVERS_FULL')
        self.rivers_tif = Config.rivers('RIVERS_TIF')
        self.rivers_interest = Config.rivers('RIVERS_AREA')

    def _prepare(self):
        """
        Some prepocessing needed to perform the Rivers processing. Unzip the
        zip file containing the full vector river file that include the area of
        interest. Rasterize the vector file to a tif file format.
        """
        unzip_resource(self.rivers_zip)
        clip_lines_vector(self.rivers_full, self.aoi, self.rivers_interest)
        rasterize_rivers(self.rivers_interest, self.rivers_tif)

    def process(self, lagoons):
        """
        Call to private "prepare" method to perform the preprocessing necessary
        to perform the processing rivers detection and correction. Open the
        file of river in tif format. Process rivers detection and correction.
        Remove intersection between rivers and lagoons

        Returns
        -------
        ndarray
            SRTM DEM image with Fourier and groves corrected cropped
            corresponding to area of interest.
        """
        self._prepare()
        rivers = gdal.Open(self.rivers_tif).ReadAsArray()
        rivers_routed = ProcessRivers(lagoons.hsheds_nan_fixed).apply(rivers)
        return ClipLagoonsRivers(lagoons.mask_lagoons,
                                 rivers_routed).apply(rivers_routed)
