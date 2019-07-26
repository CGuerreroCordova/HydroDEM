__author__ = "Cristian Guerrero Cordova"
__copyright__ = "Copyright 2018"
__credits__ = ["Cristian Guerrero Cordova"]
__version__ = "0.1"
__email__ = "cguerrerocordova@gmail.com"
__status__ = "Developing"

import numpy as np
from osgeo import gdal
from scipy import ndimage
import utilsDEM as utDEM
from settings import *


class HydroDEMProcess(object):

    def __init__(self):
        """
        class constructor
        """

    def start(self):
        """
        starts Report generation
        :param cmd_line: unix command line
        :type cmd_line: list[str]
        """

        print "Converting HSHEDS ADF to TIF file."
        # utDEM.uncompress_zip_file(HSHEDS_FILE_INPUT_ZIP)
        adf_to_tif_command = \
            GDAL_TRANSLATE + " " + HSHEDS_FILE_INPUT + " -of GTIFF " + \
            HSHEDS_FILE_TIFF
        utDEM.calling_system_call(adf_to_tif_command)

        print "Getting shape file covering area of interest."
        # utDEM.uncompress_zip_file(SRTM_FILE_INPUT_ZIP)
        utDEM.get_shape_over_area(SRTM_FILE_INPUT, SHAPE_AREA_INTEREST_INPUT,
                                  SHAPE_AREA_INTEREST_OVER)
        print "Resampling and cutting TREE file."
        # utDEM.uncompress_zip_file(TREE_CLASS_INPUT_ZIP)
        utDEM.resample_and_cut(TREE_CLASS_INPUT, SHAPE_AREA_INTEREST_OVER,
                               TREE_CLASS_AREA)
        print "Resampling and cutting SRTM file."
        utDEM.resample_and_cut(SRTM_FILE_INPUT, SHAPE_AREA_INTEREST_OVER,
                               SRTM_AREA_INTEREST_OVER)
        print "Resampling and cutting HSHEDS file."
        utDEM.resample_and_cut(HSHEDS_FILE_TIFF, SHAPE_AREA_INTEREST_OVER,
                               HSHEDS_AREA_INTEREST_OVER)

        print "Detecting and applying Fourier"
        srtm_fourier = utDEM.detect_apply_fourier(SRTM_AREA_INTEREST_OVER)

        utDEM.array2raster(HSHEDS_AREA_INTEREST_OVER,
                           "../resources/images/01_STRM_Fourier.tif",
                           srtm_fourier)

        # srtm_fourier = gdal.Open(
        #     "../resources/images/05CorrectedFourierSRTM.tif").ReadAsArray()
        print "Processing SRTM."
        print "Processing SRTM: First Iteration."
        srtm_proc1 = utDEM.process_srtm(srtm_fourier, TREE_CLASS_AREA)
        utDEM.array2raster(HSHEDS_AREA_INTEREST_OVER,
                           "../resources/images/02_STRM_Trees_First_Iteration.tif",
                           srtm_proc1)
        print "Processing SRTM: Second Iteration."
        srtm_proc2 = utDEM.process_srtm(srtm_proc1, TREE_CLASS_AREA)
        utDEM.array2raster(HSHEDS_AREA_INTEREST_OVER,
                           "../resources/images/03_STRM_Trees_Second_Iteration.tif",
                           srtm_proc2)
        print "Processing SRTM: Third Iteration."
        srtm_proc = utDEM.process_srtm(srtm_proc2, TREE_CLASS_AREA)
        utDEM.array2raster(HSHEDS_AREA_INTEREST_OVER,
                           "../resources/images/04_STRM_Trees_Third_Iteration.tif",
                           srtm_proc)
        print "Processing HSHEDS."
        hydro_sheds = gdal.Open(HSHEDS_AREA_INTEREST_OVER).ReadAsArray()
        hydro_sheds_corrected_nan = utDEM.correct_nan_values(hydro_sheds)
        print "Processing HSHEDS: Getting Lagoons."
        hsheds_mask_lagoons_values = utDEM.get_lagoons_hsheds(hydro_sheds)
        utDEM.array2raster(HSHEDS_AREA_INTEREST_OVER,
                           "../resources/images/05_Mask_Lagoon_Values.tif",
                           hsheds_mask_lagoons_values)
        hsheds_mask_lagoons = (hsheds_mask_lagoons_values > 0.0) * 1
        utDEM.array2raster(HSHEDS_AREA_INTEREST_OVER,
                           "../resources/images/06_Mask_Lagoon.tif",
                           hsheds_mask_lagoons)
        print "Processing Rivers."
        rivers_routed_closing = utDEM.process_rivers(hydro_sheds_corrected_nan,
                                                     hsheds_mask_lagoons)
        utDEM.array2raster(HSHEDS_AREA_INTEREST_OVER,
                           "../resources/images/07_Rivers_Routed.tif",
                           rivers_routed_closing)
        print "Getting terms for final merging."
        first_term_hsheds_canyons = hydro_sheds_corrected_nan * rivers_routed_closing
        utDEM.array2raster(HSHEDS_AREA_INTEREST_OVER,
                           "../resources/images/08_first_term_hsheds_canyons.tif",
                           first_term_hsheds_canyons)
        snd_term_hsheds_lagoons = hsheds_mask_lagoons_values
        utDEM.array2raster(HSHEDS_AREA_INTEREST_OVER,
                           "../resources/images/09_snd_term_hsheds_lagoons.tif",
                           snd_term_hsheds_lagoons)
        mask_canyons_lagoons = rivers_routed_closing + hsheds_mask_lagoons
        utDEM.array2raster(HSHEDS_AREA_INTEREST_OVER,
                           "../resources/images/10_mask_canyons_lagoons.tif",
                           mask_canyons_lagoons)
        not_mask_canyons_lagoons = 1 - mask_canyons_lagoons
        utDEM.array2raster(HSHEDS_AREA_INTEREST_OVER,
                           "../resources/images/11_not_mask_canyons_lagoons.tif",
                           not_mask_canyons_lagoons)
        trd_term_srtm_fourier_tree = srtm_proc * not_mask_canyons_lagoons
        utDEM.array2raster(HSHEDS_AREA_INTEREST_OVER,
                           "../resources/images/12_trd_term_srtm_fourier_tree.tif",
                           trd_term_srtm_fourier_tree)
        print "Combining parts."
        dem_ready = first_term_hsheds_canyons + snd_term_hsheds_lagoons + \
                    trd_term_srtm_fourier_tree
        print "Applying final convolve filter."
        kernel = np.ones((3, 3))
        dem_ready_convolve = ndimage.convolve(np.abs(dem_ready),
                                              weights=kernel)
        dem_ready_smooth = dem_ready_convolve / kernel.size
        utDEM.array2raster(SRTM_AREA_INTEREST_OVER, DEM_READY_SMOOTH_PATH,
                           dem_ready_smooth)
        # print "Resampling and cutting with final interest area."
        # utDEM.resample_and_cut(DEM_READY_SMOOTH_PATH,
        #                        SHAPE_AREA_INTEREST_INPUT,
        #                        DEM_READY_AREA_INTEREST)
        # print "Rotating final DEM."
        # final_dem = utDEM.rotate_and_trim(DEM_READY_AREA_INTEREST, ANGLE)
        print "Around values."
        final_dem = np.around(dem_ready_smooth)
        print "DEM Hydrologicaly conditioned ready."
        utDEM.array2raster(SRTM_AREA_INTEREST_OVER, FINAL_DEM, final_dem)
