import os, glob
from unittest import TestCase, skip
from osgeo import gdal
from numpy import testing
from cguerrero.hydrodem.hydro_dem_process import HydroDEMProcess
from cguerrero.hydrodem.utils_dem import unzip_resource
from config_loader_tests import ConfigTests


class Test_complete(TestCase):

    @classmethod
    def setUpClass(cls):
        unzip_resource(ConfigTests.resources('EXPECTED_FINAL'))

    @classmethod
    def tearDownClass(cls):
        expected_files = ConfigTests.resources('FINAL_DEM')
        files_to_delete = glob.glob(expected_files)
        for file in files_to_delete:
            os.remove(file)

    # @skip
    def test_complete_process(self):
        hydrodem_instance = HydroDEMProcess()
        final_dem_array = hydrodem_instance.start()
        final_dem_expected = gdal.Open(
            ConfigTests.resources('FINAL_DEM')).ReadAsArray()
        testing.assert_array_equal(final_dem_array, final_dem_expected)
