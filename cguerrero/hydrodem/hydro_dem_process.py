__author__ = "Cristian Guerrero Cordova"
__copyright__ = "Copyright 2019"
__credits__ = ["Cristian Guerrero Cordova"]
__version__ = "0.1"
__email__ = "cguerrerocordova@gmail.com"
__status__ = "Developing"

from io import StringIO
import cProfile, pstats
import tracemalloc
from datetime import datetime
from osgeo import gdal
import logging
from .utils_dem import (clean_workspace, unzip_resource, resample_and_cut,
                        shape_enveloping, clip_lines_vector, array2raster,
                        rasterize_rivers)
from .settings import (RIVERS_ZIP, HSHEDS_FILE_INPUT_ZIP,
                       HSHEDS_FILE_INPUT, HSHEDS_FILE_TIFF,
                       SRTM_FILE_INPUT_ZIP,
                       SRTM_FILE_INPUT, AREA_INTEREST,
                       AREA_OVER, TREE_CLASS_INPUT_ZIP,
                       TREE_CLASS_INPUT, TREE_CLASS_AREA,
                       SRTM_AREA_OVER, HSHEDS_AREA_OVER,
                       RIVERS_FULL, RIVERS_SHAPE, FINAL_DEM, PROFILE_FILE,
                       MEMORY_TIME_FILE, RIVERS_TIF)
from filters import (LagoonsDetection, ProcessRivers, ClipLagoonsRivers,
                     SRTMProcess, SubtractionFilter, ProductFilter,
                     AdditionFilter, PostProcessingFinal)

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

    def _define_shape_area(self):
        shape_enveloping(AREA_INTEREST, AREA_OVER)

    def _prepare_hsheds(self):
        unzip_resource(HSHEDS_FILE_INPUT_ZIP)
        gdt_options = gdal.TranslateOptions(format='GTIFF')
        gdal.Translate(HSHEDS_FILE_TIFF, HSHEDS_FILE_INPUT,
                       options=gdt_options)
        resample_and_cut(HSHEDS_FILE_TIFF, AREA_OVER, HSHEDS_AREA_OVER)

    def _prepare_srtm(self):
        unzip_resource(TREE_CLASS_INPUT_ZIP)
        unzip_resource(SRTM_FILE_INPUT_ZIP)
        resample_and_cut(TREE_CLASS_INPUT, AREA_OVER, TREE_CLASS_AREA)
        resample_and_cut(SRTM_FILE_INPUT, AREA_OVER, SRTM_AREA_OVER)

    def _prepare_rivers(self):
        unzip_resource(RIVERS_ZIP)
        clip_lines_vector(RIVERS_FULL, AREA_OVER, RIVERS_SHAPE)
        rasterize_rivers(RIVERS_SHAPE, RIVERS_TIF)

    def _process_srtm(self):
        srtm_raw = gdal.Open(SRTM_AREA_OVER).ReadAsArray()
        tree_class_raw = gdal.Open(TREE_CLASS_AREA).ReadAsArray()
        return SRTMProcess().apply(srtm_raw, tree_class_raw)

    def _detect_lagoons(self):
        hydro_sheds = gdal.Open(HSHEDS_AREA_OVER).ReadAsArray()
        lagoons = LagoonsDetection()
        lagoons.apply(hydro_sheds)
        return lagoons

    def _process_rivers(self, lagoons):
        rivers = gdal.Open(RIVERS_TIF).ReadAsArray()
        rivers_routed = ProcessRivers(lagoons.hsheds_nan_fixed).apply(rivers)
        return ClipLagoonsRivers(lagoons.mask_lagoons,
                                 rivers_routed).apply(rivers_routed)

    def _process_hsheds(self):
        lagoons = self._detect_lagoons()
        rivers = self._process_rivers(lagoons)
        return lagoons, rivers

    def _prepare_final_terms(self, srtm, lagoons, rivers):
        third_term_hsheds_rivers = \
            ProductFilter(factor=lagoons.hsheds_nan_fixed).apply(rivers)

        snd_term_hsheds_lagoons = lagoons.lagoons_values

        mask_rivers_lagoons = \
            AdditionFilter(adding=lagoons.mask_lagoons).apply(rivers)
        not_canyons_lagoons = \
            SubtractionFilter(minuend=1).apply(mask_rivers_lagoons)
        first_term_srtm = \
            ProductFilter(factor=srtm).apply(not_canyons_lagoons)

        return first_term_srtm, snd_term_hsheds_lagoons, third_term_hsheds_rivers

    def start(self):
        """
        starts Report generation
        :param cmd_line: unix command line
        :type cmd_line: list[str]
        """
        # Profile tool
        pr = cProfile.Profile()
        pr.enable()

        clean_workspace()

        self._define_shape_area()
        self._prepare_srtm()
        self._prepare_hsheds()
        self._prepare_rivers()
        srtm_proc = self._process_srtm()
        lagoons, rivers = self._process_hsheds()

        first_term, snd_term, third_term = self._prepare_final_terms(srtm_proc,
                                                                     lagoons,
                                                                     rivers)
        dem_complete = first_term + snd_term + third_term

        final_dem = PostProcessingFinal().apply(dem_complete)
        array2raster(FINAL_DEM, final_dem, SRTM_AREA_OVER)
        clean_workspace()

        # Profile tool
        # current, peak = tracemalloc.get_traced_memory()
        pr.disable()
        s = StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        ps.dump_stats(PROFILE_FILE.format(now))
        # with open(MEMORY_TIME_FILE.format(now), 'w') as f:
        #     f.write(f'Memory Current: {current} - Memory Peak: {peak}')
        return final_dem
