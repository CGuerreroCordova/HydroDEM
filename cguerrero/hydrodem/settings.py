import os
import configparser as conf
from os.path import dirname

BASEDIR = dirname(__file__)
config = conf.ConfigParser()
config.optionxform = str
config_ini = os.path.join(BASEDIR, 'config.ini')
config.read(config_ini)
INPUTS_FOLDER = os.path.join(BASEDIR, config['GENERAL']['INPUTS_FOLDER'])
IMAGES_FOLDER = os.path.join(BASEDIR, config['GENERAL']['IMAGES_FOLDER'])
PROFILES_FOLDER = os.path.join(BASEDIR, config['GENERAL']['PROFILES_FOLDER'])
GDAL_TRANSLATE = os.path.join(config['GENERAL']['VENV_FOLDER'],
                              config['GENERAL']['GDAL_TRANSLATE'])
SRTM_FILE_INPUT_ZIP = os.path.join(INPUTS_FOLDER,
                                   config['SRTM']['SRTM_FILE_INPUT_ZIP'])
SRTM_FILE_INPUT = os.path.join(INPUTS_FOLDER,
                               config['SRTM']['SRTM_FILE_INPUT'])
SRTM_AREA_INTEREST_OVER = os.path.join(INPUTS_FOLDER,
                                       config['SRTM'][
                                           'SRTM_AREA_INTEREST_OVER'])
HSHEDS_FILE_INPUT_ZIP = os.path.join(INPUTS_FOLDER,
                                     config['HSHEDS']['HSHEDS_FILE_INPUT_ZIP'])
HSHEDS_FILE_INPUT = os.path.join(INPUTS_FOLDER,
                                 config['HSHEDS']['HSHEDS_FILE_INPUT'])
HSHEDS_FILE_TIFF = os.path.join(INPUTS_FOLDER,
                                config['HSHEDS']['HSHEDS_FILE_TIFF'])
HSHEDS_AREA_INTEREST_OVER = os.path.join(INPUTS_FOLDER,
                                         config['HSHEDS'][
                                             'HSHEDS_AREA_INTEREST_OVER'])
SHAPE_AREA_INTEREST_INPUT = os.path.join(INPUTS_FOLDER,
                                         config['SHAPES'][
                                             'SHAPE_AREA_INTEREST_INPUT'])
SHAPE_AREA_INTEREST_OVER = os.path.join(INPUTS_FOLDER,
                                        config['SHAPES'][
                                            'SHAPE_AREA_INTEREST_OVER'])
RIVERS_SHAPE = os.path.join(INPUTS_FOLDER, config['SHAPES']['RIVERS_SHAPE'])
RIVERS_FULL = os.path.join(INPUTS_FOLDER, config['SHAPES']['RIVERS_FULL'])
RIVERS_TIF = os.path.join(INPUTS_FOLDER, config['RIVERS']['RIVERS_RASTER'])
RIVERS_ZIP = os.path.join(INPUTS_FOLDER, config['RIVERS']['RIVERS_ZIP'])
TREE_CLASS_INPUT_ZIP = os.path.join(INPUTS_FOLDER,
                                    config['TREES']['TREE_CLASS_INPUT_ZIP'])
TREE_CLASS_INPUT = os.path.join(INPUTS_FOLDER,
                                config['TREES']['TREE_CLASS_INPUT'])
TREE_CLASS_AREA = os.path.join(INPUTS_FOLDER,
                               config['TREES']['TREE_CLASS_AREA'])
DEM_READY_SMOOTH_PATH = os.path.join(IMAGES_FOLDER,
                                     config['FINAL']['DEM_READY_SMOOTH_PATH'])
DEM_READY_AREA_INTEREST = os.path.join(IMAGES_FOLDER,
                                       config['FINAL'][
                                           'DEM_READY_AREA_INTEREST'])
DEM_TEMP = os.path.join(IMAGES_FOLDER, config['TEMP']['DEM_TEMP_OVER_AREA'])
TEMP_REPROJECTED_TO_CUT = os.path.join(IMAGES_FOLDER,
                                       config['TEMP'][
                                           'TEMP_REPROJECTED_TO_CUT'])
FINAL_DEM = os.path.join(IMAGES_FOLDER, config['FINAL']['DEM_READY'])
PROFILE_FILE = os.path.join(PROFILES_FOLDER,
                            config['PROFILES']['PROFILE_FILE'])
MEMORY_TIME_FILE = os.path.join(PROFILES_FOLDER,
                                config['PROFILES']['MEMORY_AND_TIME'])
