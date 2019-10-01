import os, glob, sys
from unittest import TestCase, skip
from osgeo import gdal
from numpy import testing
from cguerrero.hydrodem.hydro_dem_process import HydroDEMProcess
from cguerrero.hydrodem.utils_dem import unzip_resource
from config_loader_tests import ConfigTests
from unittest.mock import patch
import argparse



class Test_complete(TestCase):

    @classmethod
    def setUpClass(cls):
        unzip_resource(ConfigTests.resources('COMPLETE_ZIP'))

    @classmethod
    def tearDownClass(cls):
        complete_files = ConfigTests.complete_folder() + "*"
        files_to_delete = glob.glob(complete_files)
        for file in files_to_delete:
            os.remove(file)
        os.removedirs(ConfigTests.complete_folder())

    @patch('argparse.ArgumentParser.parse_args', autospec=True)
    def test_complete_process(self, mock_arg_parser):
        mock_arg_parser.return_value = \
            argparse.Namespace(
                area_interest=ConfigTests.complete('SHAPE_AREA'))
        hydrodem_instance = HydroDEMProcess()

        final_dem_array = hydrodem_instance.start()
        final_dem_expected = gdal.Open(
            ConfigTests.complete('FINAL_DEM')).ReadAsArray()
        testing.assert_array_equal(final_dem_array, final_dem_expected)
