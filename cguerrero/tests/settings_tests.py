import os
import configparser as conf

BASEDIR = os.path.dirname(__file__)

config = conf.ConfigParser()
config.optionxform = str
config_ini = os.path.join(BASEDIR, "config_tests.ini")
config.read(config_ini)
RESOURCES_FOLDER = os.path.join(BASEDIR, config['COMMON']['RESOURCES_FOLDER'])
EXPECTED_FOLDER = os.path.join(BASEDIR, config['COMMON']['EXPECTED_FOLDER'])
OUTPUT_FOLDER = os.path.join(BASEDIR, config['COMMON']['OUTPUT_FOLDER'])

GEO_IMAGE = os.path.join(BASEDIR, RESOURCES_FOLDER,
                         config['COMMON']['GEO_IMAGE'])
OUTPUT_GEO_IMAGE = os.path.join(RESOURCES_FOLDER,
                                config['COMMON']['OUTPUT_GEO_IMAGE'])

# Test Majority Filter
HSHEDS_INPUT_MAJORITY = os.path.join(RESOURCES_FOLDER,
                                     config['INPUTS']['HSHEDS_MAJORITY'])
MAJORITY_OUTPUT = os.path.join(EXPECTED_FOLDER,
                               config['EXPECTED']['MAJORITY_FILTER'])

# Test Expand Filter
INPUT_EXPAND = os.path.join(RESOURCES_FOLDER,
                            config['INPUTS']['EXPAND_INPUT'])
EXPAND_OUTPUT = os.path.join(EXPECTED_FOLDER,
                             config['EXPECTED']['EXPAND_FILTER'])
# Test River Routing

HSHEDS_INPUT_RIVER_ROUTING = os.path.join(RESOURCES_FOLDER,
                                          config['INPUTS'][
                                              'HSHEDS_RIVER_ROUTING'])
MASK_INPUT_RIVER_ROUTING = os.path.join(RESOURCES_FOLDER,
                                        config['INPUTS'][
                                            'MASK_RIVERS_ROUTING'])
RIVER_ROUTING_EXPECTED = os.path.join(EXPECTED_FOLDER,
                                      config['EXPECTED']['RIVERS_ROUTED'])

# Test Quadratic Filter
SRTM_INPUT_QUADRATIC = os.path.join(RESOURCES_FOLDER,
                                    config['INPUTS']['SRTM_FOURIER_QUADRATIC'])

QUADRATIC_FILTER_EXPECTED = os.path.join(EXPECTED_FOLDER,
                                         config['EXPECTED'][
                                             'QUADRATIC_FILTER'])

# Test Nan Values Correction
HSHEDS_INPUT_NAN = os.path.join(RESOURCES_FOLDER,
                                config['INPUTS']['HSHEDS_INPUT_NAN'])
HSHEDS_NAN_CORRECTED = os.path.join(EXPECTED_FOLDER,
                                    config['EXPECTED']['HSHEDS_NAN_CORRECTED'])
