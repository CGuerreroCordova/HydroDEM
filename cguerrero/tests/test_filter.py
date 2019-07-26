from osgeo import gdal
from unittest import TestCase
from numpy import testing
from cguerrero.hydrodem.utilsDEM import (majority_filter, expand_filter,
                                         route_rivers)
from settings_tests import (HSHEDS_INPUT_MAJORITY, MAJORITY_OUTPUT,
                            INPUT_EXPAND, EXPAND_OUTPUT,
                            HSHEDS_INPUT_RIVER_ROUTING,
                            MASK_INPUT_RIVER_ROUTING, RIVER_ROUTING_EXPECTED)


class Test_filter(TestCase):

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
        pass
