from osgeo import gdal
from unittest import TestCase
from cguerrero.hydrodem.utilsDEM import *
import filecmp
from numpy import testing


class TestProcessDEM(TestCase):

    def test_array2raster(self):
        path_image = "cguerrero/resources/tests/geoTest.tif"
        path_new_image = "cguerrero/resources/tests/newGeoTest.tif"

        raster = gdal.Open(path_image)
        array = raster.ReadAsArray()
        array2raster(path_image, path_new_image, array)

        raster_new = gdal.Open(path_new_image)
        array_new = raster_new.ReadAsArray()

        testing.assert_array_equal(array, array_new)
        self.assertEqual(raster.GetGeoTransform(),
                         raster_new.GetGeoTransform())

    def test_array2raster_simple(self):
        path_image = "cguerrero/resources/tests/geoTest.tif"
        path_new_image = "cguerrero/resources/tests/newGeoTest.tif"

        raster = gdal.Open(path_image)
        array = raster.ReadAsArray()
        array2raster_simple(path_new_image, array)

        raster_new = gdal.Open(path_new_image)
        array_new = raster_new.ReadAsArray()

        testing.assert_array_equal(array, array_new)

    def test_get_lagoons_hsheds(self):
        path_image = "cguerrero/resources/tests/hsheds_test.tif"
        path_lagoons_test = "cguerrero/resources/tests/lagoons_test.tif"
        path_new_image = "cguerrero/resources/tests/new_lagoons_test.tif"

        raster = gdal.Open(path_image)
        array = raster.ReadAsArray()

        lagoons = get_lagoons_hsheds(array)

        array2raster_simple(path_new_image, lagoons)

        raster_new = gdal.Open(path_new_image)
        array_new = raster_new.ReadAsArray()

        raster_test = gdal.Open(path_lagoons_test)
        array_test = raster_test.ReadAsArray()

        testing.assert_array_equal(array_new, array_test)
