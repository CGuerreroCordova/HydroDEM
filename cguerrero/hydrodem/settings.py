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
SRTM_ZIP = os.path.join(INPUTS_FOLDER, config['SRTM']['SRTM_ZIP'])
SRTM_TIF = os.path.join(INPUTS_FOLDER, config['SRTM']['SRTM_TIF'])
SRTM_AREA = os.path.join(INPUTS_FOLDER, config['SRTM']['SRTM_AREA'])
HSHEDS_ZIP = os.path.join(INPUTS_FOLDER, config['HSHEDS']['HSHEDS_ZIP'])
HSHEDS_ADF = os.path.join(INPUTS_FOLDER, config['HSHEDS']['HSHEDS_ADF'])
HSHEDS_TIF = os.path.join(INPUTS_FOLDER, config['HSHEDS']['HSHEDS_TIF'])
HSHEDS_AREA = os.path.join(INPUTS_FOLDER, config['HSHEDS']['HSHEDS_AREA'])
AREA_INTEREST = os.path.join(INPUTS_FOLDER, config['SHAPES']['AREA_INTEREST'])
AREA_ENVELOPE = os.path.join(INPUTS_FOLDER, config['SHAPES']['AREA_ENVELOPE'])
RIVERS_AREA = os.path.join(INPUTS_FOLDER, config['RIVERS']['RIVERS_AREA'])
RIVERS_FULL = os.path.join(INPUTS_FOLDER, config['RIVERS']['RIVERS_FULL'])
RIVERS_TIF = os.path.join(INPUTS_FOLDER, config['RIVERS']['RIVERS_TIF'])
RIVERS_ZIP = os.path.join(INPUTS_FOLDER, config['RIVERS']['RIVERS_ZIP'])
GROVES_ZIP = os.path.join(INPUTS_FOLDER, config['GROVES']['GROVES_ZIP'])
GROVES_TIF = os.path.join(INPUTS_FOLDER, config['GROVES']['GROVES_TIF'])
GROVES_AREA = os.path.join(INPUTS_FOLDER, config['GROVES']['GROVES_AREA'])
DEM_READY_SMOOTH_PATH = os.path.join(IMAGES_FOLDER,
                                     config['FINAL']['DEM_READY_SMOOTH_PATH'])
DEM_READY_AREA_INTEREST = os.path.join(IMAGES_FOLDER,
                                       config['FINAL'][
                                           'DEM_READY_AREA_INTEREST'])
TEMP_REPROJECTED_TO_CUT = os.path.join(IMAGES_FOLDER,
                                       config['TEMP'][
                                           'TEMP_REPROJECTED_TO_CUT'])
FINAL_DEM = os.path.join(IMAGES_FOLDER, config['FINAL']['DEM_READY'])
PROFILE_FILE = os.path.join(PROFILES_FOLDER,
                            config['PROFILES']['PROFILE_FILE'])
MEMORY_TIME_FILE = os.path.join(PROFILES_FOLDER,
                                config['PROFILES']['MEMORY_AND_TIME'])
