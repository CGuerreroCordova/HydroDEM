import filecmp
import glob
import os
from unittest import TestCase

import numpy as np
from numpy import testing
from osgeo import gdal

from cguerrero.hydrodem.filters import (MajorityFilter, ExpandFilter,
                                        EnrouteRivers, QuadraticFilter,
                                        CorrectNANValues, IsolatedPoints,
                                        BlanksFourier, MaskFourier,
                                        LagoonsDetection, DetectApplyFourier)
from cguerrero.hydrodem.utils_dem import (array2raster, resample_and_cut,
                                          shape_enveloping, clip_lines_vector,
                                          unzip_resource)
from config_loader_tests import ConfigTests

class Test_filter(TestCase):

    @classmethod
    def setUpClass(cls):
        unzip_resource(ConfigTests.resources('INPUTS_ZIP'))
        unzip_resource(ConfigTests.resources('EXPECTED_ZIP'))
        if not os.path.exists(ConfigTests.output_folder()):
            os.makedirs(ConfigTests.output_folder())

    @classmethod
    def tearDownClass(cls):
        input_files = ConfigTests.inputs_folder() + "*"
        expected_files = ConfigTests.expected_folder() + "*"
        output_files = ConfigTests.output_folder() + "*"
        files_to_delete = glob.glob(input_files)
        files_to_delete += glob.glob(expected_files)
        files_to_delete += glob.glob(output_files)
        for file in files_to_delete:
            os.remove(file)
        os.removedirs(ConfigTests.inputs_folder())
        os.removedirs(ConfigTests.expected_folder())
        os.removedirs(ConfigTests.output_folder())

    def test_array2raster_georeferenced(self):
        raster = gdal.Open(ConfigTests.inputs('GEO_IMAGE'))
        array = raster.ReadAsArray()
        array2raster(ConfigTests.outputs('OUTPUT_GEO_IMAGE'), array,
                     ConfigTests.inputs('GEO_IMAGE'))
        raster_new = gdal.Open(ConfigTests.outputs('OUTPUT_GEO_IMAGE'))
        array_new = raster_new.ReadAsArray()
        testing.assert_array_equal(array, array_new)
        self.assertEqual(raster.GetGeoTransform(),
                         raster_new.GetGeoTransform())

    def test_array2raster_no_georeferenced(self):
        raster = gdal.Open(ConfigTests.inputs('GEO_IMAGE'))
        array = raster.ReadAsArray()
        array2raster(ConfigTests.outputs('OUTPUT_GEO_IMAGE'), array)
        raster_new = gdal.Open(ConfigTests.outputs('OUTPUT_GEO_IMAGE'))
        array_new = raster_new.ReadAsArray()
        testing.assert_array_equal(array, array_new)

    def test_majority_filter(self):
        hsheds_input = \
            gdal.Open(ConfigTests.inputs('HSHEDS_MAJORITY')).ReadAsArray()
        result_majority_filter = \
            MajorityFilter(window_size=11).apply(hsheds_input)
        expected_majority = \
            gdal.Open(ConfigTests.expected('MAJORITY_FILTER')).ReadAsArray()
        testing.assert_array_equal(result_majority_filter, expected_majority)

    def test_expand_filter(self):
        expand_input = \
            gdal.Open(ConfigTests.inputs('EXPAND_INPUT')).ReadAsArray()
        result_expand_filter = ExpandFilter(window_size=7).apply(expand_input)
        expected_expanded = \
            gdal.Open(ConfigTests.expected('EXPAND_FILTER')).ReadAsArray()
        testing.assert_array_equal(result_expand_filter, expected_expanded)

    def test_route_rivers(self):
        hsheds_input = \
            gdal.Open(ConfigTests.inputs('HSHEDS_RIVER_ROUTING')).ReadAsArray()
        mask_rivers = \
            gdal.Open(ConfigTests.inputs('MASK_RIVERS_ROUTING')).ReadAsArray()
        rivers_routed = \
            EnrouteRivers(window_size=3, dem=hsheds_input).apply(mask_rivers)
        rivers_routed_expected = \
            gdal.Open(ConfigTests.expected('RIVERS_ROUTED')).ReadAsArray()
        testing.assert_array_equal(rivers_routed, rivers_routed_expected)

    def test_quadratic_filter(self):
        quadratic_filter_input = \
            gdal.Open(
                ConfigTests.inputs('SRTM_FOURIER_QUADRATIC')).ReadAsArray()
        result_quadratic_filter = \
            np.around(
                QuadraticFilter(window_size=15).apply(quadratic_filter_input))
        expected_quadratic = \
            np.around(gdal.Open(ConfigTests.expected('QUADRATIC_FILTER'))
                      .ReadAsArray())
        testing.assert_array_equal(result_quadratic_filter, expected_quadratic)

    def test_nan_values_filter(self):
        nan_values_input = \
            gdal.Open(ConfigTests.inputs('HSHEDS_INPUT_NAN')).ReadAsArray()
        nan_values_corrected = CorrectNANValues().apply(nan_values_input)
        expected_nan_values_corrected = \
            gdal.Open(
                ConfigTests.expected('HSHEDS_NAN_CORRECTED')).ReadAsArray()
        testing.assert_array_equal(nan_values_corrected,
                                   expected_nan_values_corrected)

    def test_isolated_filter(self):
        mask_isolated_input = \
            gdal.Open(ConfigTests.inputs('MASK_ISOLATED')).ReadAsArray()
        isolated_points_filtered = \
            IsolatedPoints(window_size=3).apply(mask_isolated_input)
        expected_isolated_points = \
            gdal.Open(
                ConfigTests.expected('MASK_ISOLATED_FILTERED')).ReadAsArray()
        testing.assert_array_equal(isolated_points_filtered,
                                   expected_isolated_points)

    def test_filter_blanks_fourier(self):
        quarter_fourier = \
            gdal.Open(ConfigTests.inputs('QUARTER_FOURIER')).ReadAsArray()
        filtered_blanks, fourier_modified = \
            BlanksFourier(window_size=55).apply(quarter_fourier)
        expected_filtered_blanks = \
            gdal.Open(ConfigTests.expected('FILTERED_FOURIER_1')).ReadAsArray()
        expected_fourier_modified = \
            gdal.Open(ConfigTests.expected('FILTERED_FOURIER_2')).ReadAsArray()
        testing.assert_array_equal(filtered_blanks, expected_filtered_blanks)
        testing.assert_array_equal(fourier_modified, expected_fourier_modified)

    def test_get_mask_fourier(self):
        first_quarter = \
            gdal.Open(ConfigTests.inputs('FIRST_QUARTER')).ReadAsArray()
        mask_fourier = MaskFourier().apply(first_quarter)
        expected_mask_fourier = \
            gdal.Open(ConfigTests.expected('FIRST_MASK_FOURIER')).ReadAsArray()
        testing.assert_array_equal(mask_fourier, expected_mask_fourier)

    def test_detect_apply_fourier(self):
        srtm_raw = gdal.Open(ConfigTests.inputs('SRTM_STRIPPED')).ReadAsArray()
        srtm_corrected = DetectApplyFourier().apply(srtm_raw)
        array2raster(ConfigTests.outputs('SRTM_CORRECTED'), srtm_corrected)
        srtm_corrected_open = \
            gdal.Open(ConfigTests.outputs('SRTM_CORRECTED')).ReadAsArray()
        srtm_expected = \
            gdal.Open(
                ConfigTests.expected('SRTM_WITHOUT_STRIPS')).ReadAsArray()
        testing.assert_array_equal(srtm_corrected_open, srtm_expected)


    def test_resample_and_cut(self):
        resample_and_cut(ConfigTests.inputs('HSHEDS_FILE_TIFF'),
                         ConfigTests.inputs('SHAPE_AREA_INTEREST_OVER'),
                         ConfigTests.outputs('HSHEDS_AREA_INTEREST_OUTPUT'))
        output_resampled = np.around(
            gdal.Open(ConfigTests.outputs(
                'HSHEDS_AREA_INTEREST_OUTPUT')).ReadAsArray())
        expected_resampled = np.around(
            gdal.Open(
                ConfigTests.expected('HSHEDS_AREA_INTEREST')).ReadAsArray())
        testing.assert_array_equal(output_resampled, expected_resampled)

    def test_get_shape_over_area(self):
        shape_enveloping(ConfigTests.inputs('SHAPE_AREA_INPUT'),
                         ConfigTests.outputs('SHAPE_AREA_OVER_CREATED'))
        self.assertTrue(filecmp.cmp(
            ConfigTests.outputs('SHAPE_AREA_OVER_CREATED'),
            ConfigTests.expected('SHAPE_AREA_OVER')))

    def test_get_lagoons(self):
        hydro_sheds = \
            gdal.Open(ConfigTests.inputs('HYDRO_SHEDS')).ReadAsArray()
        lagoons = LagoonsDetection()
        lagoons.apply(hydro_sheds)
        lagoons_detected = lagoons.results["TidyingLagoons"]
        lagoons_expected = \
            gdal.Open(ConfigTests.expected('LAGOONS_DETECTED')).ReadAsArray()
        testing.assert_array_equal(lagoons_detected, lagoons_expected)

    def test_clip_lines_vector(self):
        clip_lines_vector(ConfigTests.inputs('RIVERS_VECTOR'),
                          ConfigTests.inputs('SHAPE_AREA_INTEREST_OVER'),
                          ConfigTests.outputs('RIVERS_CLIPPED'))
        self.assertTrue(filecmp.cmp(ConfigTests.expected('RIVERS_AREA'),
                                    ConfigTests.outputs('RIVERS_CLIPPED')))

    # def test_process_rivers(self):
    #     hsheds_nan_corrected = gdal.Open(HSHEDS_NAN_CORRECTED).ReadAsArray()
    #     mask_lagoons = gdal.Open(MASK_LAGOONS).ReadAsArray()
    #     rivers = gdal.Open(RASTER_RIVERS).ReadAsArray()
    #     rivers_processed = process_rivers(hsheds_nan_corrected, mask_lagoons,
    #                                       rivers)
    #     rivers_processed_expected = \
    #         gdal.Open(RIVERS_PROCESSED).ReadAsArray()
    #     testing.assert_array_equal(rivers_processed, rivers_processed_expected)

    def test_uncompress_zip_file(self):
        unzip_resource(ConfigTests.inputs('ZIP_FILE'))
        self.assertTrue(filecmp.cmp(ConfigTests.inputs('SRTM_UNCOMPRESS'),
                                    ConfigTests.expected('SRTM_UNCOMPRESS')))
