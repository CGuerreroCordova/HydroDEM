"""
Provide the class that contains main objects and methods to perform Hydro Dem
calculation
"""

__author__ = "Cristian Guerrero Cordova"
__copyright__ = "Copyright 2019"
__credits__ = ["Cristian Guerrero Cordova"]
__version__ = "0.1"
__email__ = "cguerrerocordova@gmail.com"
__status__ = "Developing"

from functools import wraps
from io import StringIO
import cProfile
import pstats
from datetime import datetime
import logging
from utils_dem import (shape_enveloping, array2raster, clean_workspace)
from filters.custom_filters import (SubtractionFilter, ProductFilter,
                                    AdditionFilter, PostProcessingFinal)
from config_loader import Config
from arguments_manager import ArgumentsManager
from image_srtm import SRTM
from image_hsheds import HSHEDS

LOGGER = logging.getLogger(__name__)
CONSOLE_HANDLER = logging.StreamHandler()
CONSOLE_FORMAT = logging.Formatter('%(asctime)s - %(message)s',
                                   datefmt='%d-%b-%y %H:%M:%S')
CONSOLE_HANDLER.setFormatter(CONSOLE_FORMAT)
LOGGER.addHandler(CONSOLE_HANDLER)
LOGGER.setLevel(logging.DEBUG)


class HydroDEMProcess:
    """
    Represent the main porcessing object. It is composed by area of envelope,
    SRTM and HSHEDS attributes

    Attributes
    ----------
    area_envelope : str (filepath)
        Filepath to the shapefile which the Hydro DEM process will be computed.
    srtm : SRTM
        Element that represent the processing related to SRTM corrections.
    hsheds : HSHEDS
        Element that represent the processing related to HSHEDS extractions.
    """

    def __init__(self):
        """
        Set and define the attributes of HydroDEMProcess. Get area of envelope
        from config file. Create the instances of main elements to process.
        """
        self.area_envelope = Config.shapes('AREA_ENVELOPE')
        self.srtm = SRTM(self.area_envelope)
        self.hsheds = HSHEDS(self.area_envelope)

    def _prepare_final_terms(self, srtm, lagoons, rivers):
        """
        After processing SRTM and HSHEDS DEM and get results, the elements are
        combined to get the final dem ready for hydrologic purposes

        Parameters
        ----------
        srtm : ndarray
            SRTM corrcted cropped to the area of envelope region
        lagoons : ndarray
            Lagoons detected from HSHEDS DEM.
        rivers : ndarray
            Rivers coming from HSHEDS using the rivers vector file.

        Returns
        -------
        tuple(ndarray, ndarray, ndarray)
            Final terms to be combined.
        """

        mask_rivers_lagoons = \
            AdditionFilter(addend=lagoons.mask_lagoons).apply(rivers)
        not_rivers_lagoons = \
            SubtractionFilter(minuend=1).apply(mask_rivers_lagoons)
        first_term_srtm = \
            ProductFilter(factor=srtm).apply(not_rivers_lagoons)
        third_term_hsheds_rivers = \
            ProductFilter(factor=lagoons.hsheds_nan_fixed).apply(rivers)
        snd_term_hsheds_lagoons = lagoons.lagoons_values

        return first_term_srtm, snd_term_hsheds_lagoons, \
               third_term_hsheds_rivers

    def profile(func):
        """
        Decorator function to compute profile of the execution of HydroDem
        process

        Returns
        -------
        Function enveloped by the starting and finishing of computing profile.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            profile = cProfile.Profile()
            profile.enable()
            function = func(*args, **kwargs)
            # current, peak = tracemalloc.get_traced_memory()
            profile.disable()
            stream = StringIO()
            sortby = 'cumulative'
            profile_stats = \
                pstats.Stats(profile, stream=stream).sort_stats(sortby)
            now = datetime.now().strftime('%Y%m%d_%H%M%S')
            profile_stats.dump_stats(Config.profiles('PROFILE_FILE')
                                     .format(now))
            # with open(MEMORY_TIME_FILE.format(now), 'w') as f:
            #     f.write(f'Memory Current: {current} - Memory Peak: {peak}')
            return function

        return wrapper

    @profile
    def start(self):
        """
        Launch the execution of Hydro DEM process. First of all, clean the
        workspace removing old temporary files created during previous
        execution. Get arguments from command line. Get the real area of
        envelope using the shapefile provided for the user.

        References
        ----------
        Readme.md on github respository cointains a flow chart with the
        processing of Hydro DEM https://github.com/CGuerreroCordova/HydroDEM

        Returns
        -------
        final_dem : str (filepath)
            Filepath where the ready dem for Hydrologic purposes can be found
        """
        clean_workspace()
        arg_parsed = ArgumentsManager().parse()
        shape_enveloping(arg_parsed.area_interest, self.area_envelope)
        srtm = self.srtm.process()
        lagoons, rivers = self.hsheds.process()
        first_term, snd_term, third_term = self._prepare_final_terms(srtm,
                                                                     lagoons,
                                                                     rivers)
        dem_complete = first_term + snd_term + third_term
        final_dem = PostProcessingFinal().apply(dem_complete)
        array2raster(Config.final('DEM_READY'), final_dem,
                     Config.srtm('SRTM_AREA'))
        clean_workspace()
        return final_dem
