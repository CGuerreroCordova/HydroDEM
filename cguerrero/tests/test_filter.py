import numpy as np
from osgeo import gdal
import os, glob, filecmp
from unittest import TestCase
from numpy import testing
from cguerrero.hydrodem.filters import (MajorityFilter, ExpandFilter,
                                        EnrouteRivers, QuadraticFilter,
                                        CorrectNANValues, IsolatedPoints,
                                        BlanksFourier, MaskFourier,
                                        LagoonsDetection)
from cguerrero.hydrodem.utils_dem import (array2raster,
                                          detect_apply_fourier,
                                          resample_and_cut,
                                          get_shape_over_area,
                                          clip_lines_vector,
                                          process_rivers, uncompress_zip_file)
from settings_tests import (INPUTS_ZIP, EXPECTED_ZIP, HSHEDS_INPUT_MAJORITY,
                            MAJORITY_FILTER, OUTPUT_FOLDER,
                            INPUT_EXPAND, EXPAND_OUTPUT, GEO_IMAGE,
                            HSHEDS_INPUT_RIVER_ROUTING, OUTPUT_GEO_IMAGE,
                            MASK_INPUT_RIVER_ROUTING, RIVERS_ROUTED_EXPECTED,
                            SRTM_INPUT_QUADRATIC, QUADRATIC_FILTER_EXPECTED,
                            HSHEDS_INPUT_NAN, HSHEDS_NAN_CORRECTED,
                            MASK_ISOLATED, MASK_ISOLATED_FILTERED,
                            QUARTER_FOURIER, FILTERED_FOURIER_1,
                            FILTERED_FOURIER_2, FIRST_QUARTER, RASTER_RIVERS,
                            FIRST_MASK_FOURIER, SRTM_STRIPPED,
                            SRTM_WITHOUT_STRIPS, SRTM_CORRECTED, MASK_TREES,
                            SRTM_PROCESSED, HSHEDS_FILE_TIFF,
                            SHAPE_AREA_INTEREST_OVER, HSHEDS_AREA_INTEREST,
                            HSHEDS_AREA_INTEREST_OUTPUT, SHAPE_AREA_INPUT,
                            SHAPE_AREA_OVER, SRTM_UNCOMPRESS_EXPECTED,
                            SHAPE_AREA_OVER_CREATED, OUTPUT_FOLDER,
                            HYDRO_SHEDS, ZIP_FILE, SRTM_UNCOMPRESSED,
                            LAGOONS_DETECTED, RIVERS_VECTOR, RIVERS_CLIPPED,
                            RIVERS_AREA, MASK_LAGOONS, RIVERS_PROCESSED,
                            INPUTS_FOLDER, EXPECTED_FOLDER, RIVERS_AREA_INPUT,
                            SRTM_PROCESSED_OUTPUT)

class Test_filter(TestCase):

    @classmethod
    def setUpClass(cls):
        uncompress_zip_file(INPUTS_ZIP)
        uncompress_zip_file(EXPECTED_ZIP)
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

    @classmethod
    def tearDownClass(cls):
        input_files = INPUTS_FOLDER + "*"
        expected_files = EXPECTED_FOLDER + "*"
        output_files = OUTPUT_FOLDER + "*"
        files_to_delete = glob.glob(input_files)
        files_to_delete += glob.glob(expected_files)
        files_to_delete += glob.glob(output_files)
        for file in files_to_delete:
            os.remove(file)
        os.removedirs(INPUTS_FOLDER)
        os.removedirs(EXPECTED_FOLDER)
        os.removedirs(OUTPUT_FOLDER)
        # pass

    def test_array2raster_georeferenced(self):
        raster = gdal.Open(GEO_IMAGE)
        array = raster.ReadAsArray()
        array2raster(OUTPUT_GEO_IMAGE, array, GEO_IMAGE)
        raster_new = gdal.Open(OUTPUT_GEO_IMAGE)
        array_new = raster_new.ReadAsArray()
        testing.assert_array_equal(array, array_new)
        self.assertEqual(raster.GetGeoTransform(),
                         raster_new.GetGeoTransform())

    def test_array2raster_no_georeferenced(self):
        raster = gdal.Open(GEO_IMAGE)
        array = raster.ReadAsArray()
        array2raster(OUTPUT_GEO_IMAGE, array)
        raster_new = gdal.Open(OUTPUT_GEO_IMAGE)
        array_new = raster_new.ReadAsArray()
        testing.assert_array_equal(array, array_new)

    def test_majority_filter(self):
        hsheds_input = gdal.Open(HSHEDS_INPUT_MAJORITY).ReadAsArray()
        result_majority_filter = \
            MajorityFilter(window_size=11).apply(hsheds_input)
        expected_majority = gdal.Open(MAJORITY_FILTER).ReadAsArray()
        testing.assert_array_equal(result_majority_filter, expected_majority)

    def test_expand_filter(self):
        expand_input = gdal.Open(INPUT_EXPAND).ReadAsArray()
        result_expand_filter = ExpandFilter(window_size=7).apply(expand_input)
        expected_expanded = gdal.Open(EXPAND_OUTPUT).ReadAsArray()
        testing.assert_array_equal(result_expand_filter, expected_expanded)

    def test_route_rivers(self):
        hsheds_input = gdal.Open(HSHEDS_INPUT_RIVER_ROUTING).ReadAsArray()
        mask_rivers = gdal.Open(MASK_INPUT_RIVER_ROUTING).ReadAsArray()
        rivers_routed = EnrouteRivers(window_size=3, dem=hsheds_input).apply(
            mask_rivers)
        rivers_routed_expected = gdal.Open(
            RIVERS_ROUTED_EXPECTED).ReadAsArray()
        testing.assert_array_equal(rivers_routed, rivers_routed_expected)

    def test_quadratic_filter(self):
        quadratic_filter_input = gdal.Open(SRTM_INPUT_QUADRATIC).ReadAsArray()
        result_quadratic_filter = \
            np.around(
                QuadraticFilter(window_size=15).apply(quadratic_filter_input))
        expected_quadratic = np.around(
            gdal.Open(QUADRATIC_FILTER_EXPECTED).ReadAsArray())
        testing.assert_array_equal(result_quadratic_filter, expected_quadratic)

    def test_nan_values_filter(self):
        nan_values_input = gdal.Open(HSHEDS_INPUT_NAN).ReadAsArray()
        nan_values_corrected = CorrectNANValues().apply(nan_values_input)
        expected_nan_values_corrected = gdal.Open(
            HSHEDS_NAN_CORRECTED).ReadAsArray()
        testing.assert_array_equal(nan_values_corrected,
                                   expected_nan_values_corrected)

    def test_isolated_filter(self):
        mask_isolated_input = gdal.Open(MASK_ISOLATED).ReadAsArray()
        isolated_points_filtered = \
            IsolatedPoints(window_size=3).apply(mask_isolated_input)
        expected_isolated_points = \
            gdal.Open(MASK_ISOLATED_FILTERED).ReadAsArray()
        testing.assert_array_equal(isolated_points_filtered,
                                   expected_isolated_points)

    def test_filter_blanks_fourier(self):
        quarter_fourier = gdal.Open(QUARTER_FOURIER).ReadAsArray()
        filtered_blanks, fourier_modified = \
            BlanksFourier(window_size=55).apply(quarter_fourier)

        expected_filtered_blanks = gdal.Open(FILTERED_FOURIER_1).ReadAsArray()
        expected_fourier_modified = gdal.Open(FILTERED_FOURIER_2).ReadAsArray()
        testing.assert_array_equal(filtered_blanks, expected_filtered_blanks)
        testing.assert_array_equal(fourier_modified, expected_fourier_modified)

    def test_get_mask_fourier(self):
        first_quarter = gdal.Open(FIRST_QUARTER).ReadAsArray()
        mask_fourier = MaskFourier().apply(first_quarter)
        expected_mask_fourier = gdal.Open(FIRST_MASK_FOURIER).ReadAsArray()
        testing.assert_array_equal(mask_fourier, expected_mask_fourier)

    def test_detect_apply_fourier(self):
        srtm_raw = gdal.Open(SRTM_STRIPPED).ReadAsArray()
        srtm_corrected = detect_apply_fourier(srtm_raw)
        array2raster(SRTM_CORRECTED, srtm_corrected)
        srtm_corrected_open = gdal.Open(SRTM_CORRECTED).ReadAsArray()
        srtm_expected = gdal.Open(SRTM_WITHOUT_STRIPS).ReadAsArray()
        testing.assert_array_equal(srtm_corrected_open, srtm_expected)


    def test_resample_and_cut(self):
        resample_and_cut(HSHEDS_FILE_TIFF, SHAPE_AREA_INTEREST_OVER,
                         HSHEDS_AREA_INTEREST_OUTPUT)
        output_resampled = np.around(
            gdal.Open(HSHEDS_AREA_INTEREST_OUTPUT).ReadAsArray())
        expected_resampled = np.around(
            gdal.Open(HSHEDS_AREA_INTEREST).ReadAsArray())
        testing.assert_array_equal(output_resampled, expected_resampled)

    def test_get_shape_over_area(self):
        get_shape_over_area(SHAPE_AREA_INPUT, SHAPE_AREA_OVER_CREATED)
        self.assertTrue(filecmp.cmp(SHAPE_AREA_OVER_CREATED, SHAPE_AREA_OVER))

    def test_get_lagoons(self):
        hydro_sheds = gdal.Open(HYDRO_SHEDS).ReadAsArray()
        lagoons_detected = LagoonsDetection().apply(hydro_sheds)
        lagoons_expected = gdal.Open(LAGOONS_DETECTED).ReadAsArray()
        testing.assert_array_equal(lagoons_detected, lagoons_expected)

    def test_clip_lines_vector(self):
        clip_lines_vector(RIVERS_VECTOR, SHAPE_AREA_INTEREST_OVER,
                          RIVERS_CLIPPED)
        self.assertTrue(filecmp.cmp(RIVERS_AREA, RIVERS_CLIPPED))

    def test_process_rivers(self):
        hsheds_nan_corrected = gdal.Open(HSHEDS_NAN_CORRECTED).ReadAsArray()
        mask_lagoons = gdal.Open(MASK_LAGOONS).ReadAsArray()
        rivers = gdal.Open(RASTER_RIVERS).ReadAsArray()
        rivers_processed = process_rivers(hsheds_nan_corrected, mask_lagoons,
                                          rivers)
        rivers_processed_expected = \
            gdal.Open(RIVERS_PROCESSED).ReadAsArray()
        testing.assert_array_equal(rivers_processed, rivers_processed_expected)

    def test_uncompress_zip_file(self):
        uncompress_zip_file(ZIP_FILE)
        self.assertTrue(filecmp.cmp(SRTM_UNCOMPRESSED,
                                    SRTM_UNCOMPRESS_EXPECTED))
