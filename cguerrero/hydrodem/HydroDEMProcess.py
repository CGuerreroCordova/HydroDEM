__author__ = "Cristian Guerrero Cordova"
__copyright__ = "Copyright 2018"
__credits__ = ["Cristian Guerrero Cordova"]
__version__ = "0.1"
__email__ = "cguerrerocordova@gmail.com"
__status__ = "Developing"

import os
from io import StringIO
import cProfile, pstats
import tracemalloc
from datetime import datetime
import numpy as np
from osgeo import gdal
from scipy import ndimage
from .utils_dem import (clean_workspace, uncompress_zip_file, resample_and_cut,
                        get_shape_over_area, detect_apply_fourier,
                        process_srtm, correct_nan_values, get_lagoons_hsheds,
                        clip_lines_vector, process_rivers, array2raster,
                        array2raster_simple)
from .settings import (RIVERS_ZIP, HSHEDS_FILE_INPUT_ZIP,
                       HSHEDS_FILE_INPUT, HSHEDS_FILE_TIFF, SRTM_FILE_INPUT_ZIP,
                       SRTM_FILE_INPUT, SHAPE_AREA_INTEREST_INPUT,
                       SHAPE_AREA_INTEREST_OVER, TREE_CLASS_INPUT_ZIP,
                       TREE_CLASS_INPUT, TREE_CLASS_AREA,
                       SRTM_AREA_INTEREST_OVER, HSHEDS_AREA_INTEREST_OVER,
                       RIVERS_FULL, RIVERS_SHAPE, FINAL_DEM, PROFILE_FILE,
                       MEMORY_TIME_FILE)


class HydroDEMProcess(object):

    def __init__(self):
        """
        class constructor
        """
        pass

    def start(self):
        """
        starts Report generation
        :param cmd_line: unix command line
        :type cmd_line: list[str]
        """
        # Profile tool
        pr = cProfile.Profile()
        pr.enable()

        # The processing
        print("Cleaning Workspace")
        clean_workspace()
        print("Converting HSHEDS ADF to TIF file.")
        uncompress_zip_file(HSHEDS_FILE_INPUT_ZIP)
        gdt_options = gdal.TranslateOptions(format='GTIFF')
        gdal.Translate(HSHEDS_FILE_TIFF, HSHEDS_FILE_INPUT,
                       options=gdt_options)
        print("Getting shape file covering area of interest.")
        get_shape_over_area(SHAPE_AREA_INTEREST_INPUT,
                            SHAPE_AREA_INTEREST_OVER)
        print("Resampling and cutting TREE file.")
        uncompress_zip_file(TREE_CLASS_INPUT_ZIP)
        resample_and_cut(TREE_CLASS_INPUT, SHAPE_AREA_INTEREST_OVER,
                         TREE_CLASS_AREA)
        print("Resampling and cutting SRTM file.")
        uncompress_zip_file(SRTM_FILE_INPUT_ZIP)
        resample_and_cut(SRTM_FILE_INPUT, SHAPE_AREA_INTEREST_OVER,
                         SRTM_AREA_INTEREST_OVER)
        print("Resampling and cutting HSHEDS file.")
        resample_and_cut(HSHEDS_FILE_TIFF, SHAPE_AREA_INTEREST_OVER,
                         HSHEDS_AREA_INTEREST_OVER)
        print("Detecting and applying Fourier")
        # tracemalloc.start()
        srtm_fourier = detect_apply_fourier(SRTM_AREA_INTEREST_OVER)
        print("Processing SRTM.")
        print("Processing SRTM: First Iteration.")
        srtm_proc1 = process_srtm(srtm_fourier, TREE_CLASS_AREA)
        print("Processing SRTM: Second Iteration.")
        srtm_proc2 = process_srtm(srtm_proc1, TREE_CLASS_AREA)
        print("Processing SRTM: Third Iteration.")
        srtm_proc = process_srtm(srtm_proc2, TREE_CLASS_AREA)
        print("Processing HSHEDS.")
        hydro_sheds = gdal.Open(HSHEDS_AREA_INTEREST_OVER).ReadAsArray()
        hydro_sheds_corrected_nan = correct_nan_values(hydro_sheds)
        print("Processing HSHEDS: Getting Lagoons.")
        hsheds_mask_lagoons_values = get_lagoons_hsheds(hydro_sheds)
        hsheds_mask_lagoons = (hsheds_mask_lagoons_values > 0.0) * 1
        print("Processing Rivers.")
        uncompress_zip_file(RIVERS_ZIP)
        clip_lines_vector(RIVERS_FULL, SHAPE_AREA_INTEREST_OVER, RIVERS_SHAPE)
        rivers_routed_closing = process_rivers(hydro_sheds_corrected_nan,
                                               hsheds_mask_lagoons,
                                               RIVERS_SHAPE)
        print("Getting terms for final merging.")
        first_term_hsheds_canyons = hydro_sheds_corrected_nan * \
                                    rivers_routed_closing
        snd_term_hsheds_lagoons = hsheds_mask_lagoons_values
        mask_canyons_lagoons = rivers_routed_closing + hsheds_mask_lagoons
        not_mask_canyons_lagoons = 1 - mask_canyons_lagoons
        trd_term_srtm_fourier_tree = srtm_proc * not_mask_canyons_lagoons
        print("Combining parts.")
        dem_ready = first_term_hsheds_canyons + snd_term_hsheds_lagoons + \
                    trd_term_srtm_fourier_tree
        print("Applying final convolve filter.")
        kernel = np.ones((3, 3))
        dem_ready_convolve = ndimage.convolve(np.abs(dem_ready),
                                              weights=kernel)
        dem_ready_smooth = dem_ready_convolve / kernel.size
        print("Around values.")
        final_dem = np.around(dem_ready_smooth)
        print("DEM Hydrologicaly conditioned ready.")
        print("DEM Ready to use can be found at {}.".format(FINAL_DEM))
        array2raster(SRTM_AREA_INTEREST_OVER, FINAL_DEM, final_dem)
        print("Cleaning Workspace")
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
