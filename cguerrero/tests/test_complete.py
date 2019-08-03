import os, glob
from unittest import TestCase
from osgeo import gdal
from numpy import testing
from cguerrero.hydrodem.HydroDEMProcess import HydroDEMProcess
from cguerrero.hydrodem.utils_dem import uncompress_zip_file
from .settings_tests import FINAL_DEM_ZIP, FINAL_DEM_TEST


class Test_complete(TestCase):

    @classmethod
    def setUpClass(cls):
        uncompress_zip_file(FINAL_DEM_ZIP)

    @classmethod
    def tearDownClass(cls):
        expected_files = FINAL_DEM_TEST
        files_to_delete = glob.glob(expected_files)
        for file in files_to_delete:
            os.remove(file)

    def test_complete_process(self):
        hydrodem_instance = HydroDEMProcess()
        final_dem_array = hydrodem_instance.start()
        final_dem_expected = gdal.Open(FINAL_DEM_TEST).ReadAsArray()
        testing.assert_array_equal(final_dem_array, final_dem_expected)
