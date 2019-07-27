import numpy as np
from osgeo import gdal
import os, glob, filecmp
from unittest import TestCase
from numpy import testing
from cguerrero.hydrodem.utilsDEM import (majority_filter, expand_filter,
                                         route_rivers, quadratic_filter,
                                         correct_nan_values,
                                         filter_isolated_pixels, filter_blanks,
                                         get_mask_fourier,
                                         detect_apply_fourier,
                                         array2raster_simple, process_srtm,
                                         resample_and_cut, get_shape_over_area,
                                         get_lagoons_hsheds, clip_lines_vector,
                                         process_rivers)
from settings_tests import (HSHEDS_INPUT_MAJORITY, MAJORITY_OUTPUT,
                            INPUT_EXPAND, EXPAND_OUTPUT,
                            HSHEDS_INPUT_RIVER_ROUTING,
                            MASK_INPUT_RIVER_ROUTING, RIVER_ROUTING_EXPECTED,
                            SRTM_INPUT_QUADRATIC, QUADRATIC_FILTER_EXPECTED,
                            HSHEDS_INPUT_NAN, HSHEDS_NAN_CORRECTED,
                            MASK_ISOLATED, MASK_ISOLATED_FILTERED,
                            QUARTER_FOURIER, FILTERED_FOURIER_1,
                            FILTERED_FOURIER_2, FIRST_QUARTER,
                            FIRST_MASK_FOURIER, SRTM_STRIPPED,
                            SRTM_WITHOUT_STRIPS, SRTM_CORRECTED, MASK_TREES,
                            SRTM_PROCESSED, HSHEDS_FILE_TIFF,
                            SHAPE_AREA_INTEREST_OVER, HSHEDS_AREA_INTEREST,
                            HSHEDS_AREA_INTEREST_OUTPUT, IMAGE_TEMP,
                            SHAPE_AREA_OVER,
                            SHAPE_AREA_OVER_CREATED, OUTPUT_FOLDER,
                            HYDRO_SHEDS,
                            LAGOONS_DETECTED, RIVERS_VECTOR, RIVERS_CLIPPED,
                            RIVERS_AREA, MASK_LAGOONS, RIVERS_ROUTED_CLOSING)

class Test_filter(TestCase):

    def tearDown(self):
        file_list = glob.glob(OUTPUT_FOLDER + "*")
        for file_path in file_list:
            try:
                os.remove(file_path)
            except OSError:
                print("Error while deleting file : ", file_path)
        # pass

    def test_majority_filter(self):
        hsheds_input = gdal.Open(HSHEDS_INPUT_MAJORITY).ReadAsArray()
        result_majority_filter = majority_filter(hsheds_input, 11)
        expected_majority = gdal.Open(MAJORITY_OUTPUT).ReadAsArray()
        testing.assert_array_equal(result_majority_filter, expected_majority)

    def test_expand_filter(self):
        expand_input = gdal.Open(INPUT_EXPAND).ReadAsArray()
        result_expand_filter = expand_filter(expand_input, 7)
        expected_expanded = gdal.Open(EXPAND_OUTPUT).ReadAsArray()
        testing.assert_array_equal(result_expand_filter, expected_expanded)

    def test_route_rivers(self):
        hsheds_input = gdal.Open(HSHEDS_INPUT_RIVER_ROUTING).ReadAsArray()
        mask_rivers = gdal.Open(MASK_INPUT_RIVER_ROUTING).ReadAsArray()
        result_route_rivers = route_rivers(hsheds_input, mask_rivers, 3)
        expected_routing = gdal.Open(RIVER_ROUTING_EXPECTED).ReadAsArray()
        testing.assert_array_equal(result_route_rivers, expected_routing)

    def test_quadratic_filter(self):
        quadratic_filter_input = gdal.Open(SRTM_INPUT_QUADRATIC).ReadAsArray()
        result_quadratic_filter = np.around(
            quadratic_filter(quadratic_filter_input))
        expected_quadratic = np.around(
            gdal.Open(QUADRATIC_FILTER_EXPECTED).ReadAsArray())
        testing.assert_array_equal(result_quadratic_filter, expected_quadratic)

    def test_nan_values_filter(self):
        nan_values_input = gdal.Open(HSHEDS_INPUT_NAN).ReadAsArray()
        nan_values_corrected = correct_nan_values(nan_values_input)
        expected_nan_values_corrected = gdal.Open(
            HSHEDS_NAN_CORRECTED).ReadAsArray()
        testing.assert_array_equal(nan_values_corrected,
                                   expected_nan_values_corrected)

    def test_isolated_filter(self):
        mask_isolated_input = gdal.Open(MASK_ISOLATED).ReadAsArray()
        isolated_points_filtered = filter_isolated_pixels(mask_isolated_input,
                                                          3)
        expected_isolated_points = \
            gdal.Open(MASK_ISOLATED_FILTERED).ReadAsArray()
        testing.assert_array_equal(isolated_points_filtered,
                                   expected_isolated_points)

    def test_filter_blanks_fourier(self):
        quarter_fourier = gdal.Open(QUARTER_FOURIER).ReadAsArray()
        filtered_blanks, fourier_modified = filter_blanks(quarter_fourier, 55)
        expected_filtered_blanks = gdal.Open(FILTERED_FOURIER_1).ReadAsArray()
        expected_fourier_modified = gdal.Open(FILTERED_FOURIER_2).ReadAsArray()
        testing.assert_array_equal(filtered_blanks, expected_filtered_blanks)
        testing.assert_array_equal(fourier_modified, expected_fourier_modified)

    def test_get_mask_fourier(self):
        first_quarter = gdal.Open(FIRST_QUARTER).ReadAsArray()
        mask_fourier = get_mask_fourier(first_quarter)
        expected_mask_fourier = gdal.Open(FIRST_MASK_FOURIER).ReadAsArray()
        testing.assert_array_equal(mask_fourier, expected_mask_fourier)

    def test_detect_apply_fourier(self):
        srtm_corrected = detect_apply_fourier(SRTM_STRIPPED)
        array2raster_simple(SRTM_CORRECTED, srtm_corrected)
        srtm_corrected_open = gdal.Open(SRTM_CORRECTED).ReadAsArray()
        srtm_expected = gdal.Open(SRTM_WITHOUT_STRIPS).ReadAsArray()
        testing.assert_array_equal(srtm_corrected_open, srtm_expected)

    def test_process_srtm(self):
        srtm_to_process = gdal.Open(SRTM_WITHOUT_STRIPS).ReadAsArray()
        srtm_processed = np.around(process_srtm(srtm_to_process, MASK_TREES))
        srtm_expected = np.around(gdal.Open(SRTM_PROCESSED).ReadAsArray())
        testing.assert_array_equal(srtm_processed, srtm_expected)

    def test_resample_and_cut(self):
        # TODO: Give an smaller image of input
        resample_and_cut(HSHEDS_FILE_TIFF, SHAPE_AREA_INTEREST_OVER,
                         HSHEDS_AREA_INTEREST_OUTPUT)
        output_resampled = gdal.Open(HSHEDS_AREA_INTEREST_OUTPUT).ReadAsArray()
        expected_resampled = gdal.Open(HSHEDS_AREA_INTEREST).ReadAsArray()
        testing.assert_array_equal(output_resampled, expected_resampled)

    def test_get_shape_over_area(self):
        get_shape_over_area(IMAGE_TEMP, SHAPE_AREA_OVER_CREATED)
        self.assertTrue(filecmp.cmp(SHAPE_AREA_OVER_CREATED, SHAPE_AREA_OVER))

    def test_get_lagoons(self):
        hydro_sheds = gdal.Open(HYDRO_SHEDS).ReadAsArray()
        lagoons_detected = get_lagoons_hsheds(hydro_sheds)
        lagoons_expected = gdal.Open(LAGOONS_DETECTED).ReadAsArray()
        testing.assert_array_equal(lagoons_detected, lagoons_expected)

    def test_clip_lines_vector(self):
        clip_lines_vector(RIVERS_VECTOR, SHAPE_AREA_INTEREST_OVER,
                          RIVERS_CLIPPED)
        self.assertTrue(filecmp.cmp(RIVERS_AREA, RIVERS_CLIPPED))

    def test_process_rivers(self):
        hsheds_nan_corrected = gdal.Open(HSHEDS_NAN_CORRECTED).ReadAsArray()
        mask_lagoons = gdal.Open(MASK_LAGOONS).ReadAsArray()
        rivers_routed_closing = process_rivers(hsheds_nan_corrected,
                                               mask_lagoons, RIVERS_AREA)
        rivers_routed_closing_expected = \
            gdal.Open(RIVERS_ROUTED_CLOSING).ReadAsArray()
        testing.assert_array_equal(rivers_routed_closing,
                                   rivers_routed_closing_expected)
