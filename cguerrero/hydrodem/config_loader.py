import os
from functools import wraps
from configparser import ConfigParser

class Config:
    """Interact with configuration variables."""

    config_parser = ConfigParser()
    config_parser.optionxform = str
    config_ini = os.path.join(os.getcwd(), 'config.ini')
    config_parser.read(config_ini)

    def add_folder(folder):
        def inner_function(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                folders = {'inputs': Config._input_folder,
                           'images': Config._images_folder,
                           'profiles': Config._profiles_folder,
                           'simulation': Config._simulation_folder}
                return os.path.join(os.getcwd(), folders[folder](),
                                    func(*args, **kwargs))
            return wrapper
        return inner_function

    @classmethod
    def initialize(cls, config_file=None):
        """To read an specific config file"""
        if config_file:
            cls.config_parser.read(config_file)

    @classmethod
    def _input_folder(cls):
        return cls.config_parser.get('FOLDERS', 'INPUTS_FOLDER')

    @classmethod
    def _images_folder(cls):
        return cls.config_parser.get('FOLDERS', 'IMAGES_FOLDER')

    @classmethod
    def _profiles_folder(cls):
        return cls.config_parser.get('FOLDERS', 'PROFILES_FOLDER')

    @classmethod
    def _simulation_folder(cls):
        return cls.config_parser.get('FOLDERS', 'SIMULATION_FOLDER')


    @classmethod
    @add_folder('inputs')
    def srtm(cls, key):
        """Get srtm values from config.ini."""
        return cls.config_parser.get('SRTM', key)

    @classmethod
    @add_folder('inputs')
    def hsheds(cls, key):
        """Get hsheds values from config.ini."""
        return cls.config_parser.get('HSHEDS', key)

    @classmethod
    @add_folder('inputs')
    def rivers(cls, key):
        return cls.config_parser.get('RIVERS', key)

    @classmethod
    @add_folder('inputs')
    def shapes(cls, key):
        """Get shapes values from config.ini."""
        return cls.config_parser.get('SHAPES', key)

    @classmethod
    @add_folder('inputs')
    def groves(cls, key):
        """Get groves values from config.ini."""
        return cls.config_parser.get('GROVES', key)

    @classmethod
    @add_folder('images')
    def final(cls, key):
        """Get final values from config.ini."""
        return cls.config_parser.get('FINAL', key)

    @classmethod
    @add_folder('images')
    def temp(cls, key):
        """Get temp values from config.ini."""
        return cls.config_parser.get('TEMP', key)

    @classmethod
    @add_folder('profiles')
    def profiles(cls, key):
        """Get profiles values from config.ini."""
        return cls.config_parser.get('PROFILES', key)

    @classmethod
    @add_folder('simulation')
    def simulation(cls, key):
        """Get srtm values from config.ini."""
        return cls.config_parser.get('SIMULATION', key)
