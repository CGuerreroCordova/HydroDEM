from images import SRTM, HSHEDS
from functools import wraps

__author__ = "Cristian Guerrero Cordova"
__copyright__ = "Copyright 2019"
__credits__ = ["Cristian Guerrero Cordova"]
__version__ = "0.1"
__email__ = "cguerrerocordova@gmail.com"
__status__ = "Developing"

from io import StringIO
import cProfile, pstats
from datetime import datetime
import logging
from .utils_dem import (shape_enveloping, array2raster, clean_workspace)
from filters import (SubtractionFilter, ProductFilter, AdditionFilter,
                     PostProcessingFinal)
from .config_loader import Config
from arguments_manager import ArgumentsManager

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_format = logging.Formatter('%(asctime)s - %(message)s',
                                   datefmt='%d-%b-%y %H:%M:%S')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)


class HydroDEMProcess(object):

    def __init__(self):
        """
        class constructor
        """

        pass

    def _define_shape_area(self, area_of_interest):
        area_envelope = shape_enveloping(area_of_interest,
                                         Config.shapes('AREA_ENVELOPE'))
        return area_envelope

    def _prepare_final_terms(self, srtm, lagoons, rivers):
        mask_rivers_lagoons = \
            AdditionFilter(adding=lagoons.mask_lagoons).apply(rivers)
        not_canyons_lagoons = \
            SubtractionFilter(minuend=1).apply(mask_rivers_lagoons)
        first_term_srtm = \
            ProductFilter(factor=srtm).apply(not_canyons_lagoons)
        third_term_hsheds_rivers = \
            ProductFilter(factor=lagoons.hsheds_nan_fixed).apply(rivers)
        snd_term_hsheds_lagoons = lagoons.lagoons_values

        return first_term_srtm, snd_term_hsheds_lagoons, \
               third_term_hsheds_rivers

    def profile(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            f = func(*args, **kwargs)
            # current, peak = tracemalloc.get_traced_memory()
            pr.disable()
            s = StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            now = datetime.now().strftime('%Y%m%d_%H%M%S')
            ps.dump_stats(Config.profiles('PROFILE_FILE').format(now))
            # with open(MEMORY_TIME_FILE.format(now), 'w') as f:
            #     f.write(f'Memory Current: {current} - Memory Peak: {peak}')
            return f

        return wrapper

    @profile
    def start(self):
        """
        starts Hydro DEM Process
        :param cmd_line: unix command line
        :type cmd_line: list[str]
        """
        clean_workspace()
        arg_parsed = ArgumentsManager().parse()
        area_envelope = self._define_shape_area(arg_parsed.area_interest)
        srtm_proc = SRTM(area_envelope).process()
        lagoons, rivers = HSHEDS(area_envelope).process()
        first_term, snd_term, third_term = self._prepare_final_terms(srtm_proc,
                                                                     lagoons,
                                                                     rivers)
        dem_complete = first_term + snd_term + third_term
        final_dem = PostProcessingFinal().apply(dem_complete)
        array2raster(Config.final('DEM_READY'), final_dem,
                     Config.srtm('SRTM_AREA'))
        clean_workspace()
        return final_dem
