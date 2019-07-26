from osgeo import gdal
from unittest import TestCase
from cguerrero.hydrodem.utilsDEM import (array2raster, array2raster_simple)
import filecmp
from numpy import testing
from settings_tests import GEO_IMAGE, OUTPUT_GEO_IMAGE


class TestUtilsDEM(TestCase):

    def test_array2raster(self):
        raster = gdal.Open(GEO_IMAGE)
        array = raster.ReadAsArray()
        array2raster(GEO_IMAGE, OUTPUT_GEO_IMAGE, array)

        raster_new = gdal.Open(OUTPUT_GEO_IMAGE)
        array_new = raster_new.ReadAsArray()

        testing.assert_array_equal(array, array_new)
        self.assertEqual(raster.GetGeoTransform(),
                         raster_new.GetGeoTransform())

    def test_array2raster_simple(self):
        raster = gdal.Open(GEO_IMAGE)
        array = raster.ReadAsArray()
        array2raster_simple(OUTPUT_GEO_IMAGE, array)

        raster_new = gdal.Open(OUTPUT_GEO_IMAGE)
        array_new = raster_new.ReadAsArray()

        testing.assert_array_equal(array, array_new)


