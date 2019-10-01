import os
from functools import wraps
from configparser import ConfigParser


class ConfigTests:
    """Interact with configuration variables."""
    BASEDIR = os.path.dirname(__file__)
    config_parser = ConfigParser()
    config_parser.optionxform = str
    config_ini = os.path.join(BASEDIR, 'config_tests.ini')
    config_parser.read(config_ini)

    @classmethod
    def initialize(cls, config_file=None):
        """To read an specific config file"""
        if config_file:
            cls.config_parser.read(config_file)

    def add_folder(folder):
        def inner_function(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                folders = {'resources': ConfigTests._resources_folder,
                           'inputs': ConfigTests.inputs_folder,
                           'expected': ConfigTests.expected_folder,
                           'output': ConfigTests.output_folder,
                           'complete': ConfigTests.complete_folder}
                return os.path.join(folders[folder](), func(*args, **kwargs))
            return wrapper
        return inner_function

    def add_base_folder(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return os.path.join(ConfigTests.BASEDIR, func(*args, **kwargs))
        return wrapper

    @classmethod
    @add_base_folder
    def _resources_folder(cls):
        return cls.config_parser.get('FOLDERS', 'RESOURCES_FOLDER')

    @classmethod
    @add_base_folder
    def inputs_folder(cls):
        return cls.config_parser.get('FOLDERS', 'INPUTS_FOLDER')

    @classmethod
    @add_base_folder
    def expected_folder(cls):
        return cls.config_parser.get('FOLDERS', 'EXPECTED_FOLDER')

    @classmethod
    @add_base_folder
    def output_folder(cls):
        return cls.config_parser.get('FOLDERS', 'OUTPUT_FOLDER')

    @classmethod
    @add_base_folder
    def complete_folder(cls):
        return cls.config_parser.get('FOLDERS', 'COMPLETE_FOLDER')

    @classmethod
    @add_folder('resources')
    def resources(cls, key):
        return cls.config_parser.get('RESOURCES', key)

    @classmethod
    @add_folder('inputs')
    def inputs(cls, key):
        return cls.config_parser.get('INPUTS', key)

    @classmethod
    @add_folder('expected')
    def expected(cls, key):
        return cls.config_parser.get('EXPECTED', key)

    @classmethod
    @add_folder('output')
    def outputs(cls, key):
        return cls.config_parser.get('OUTPUTS', key)

    @classmethod
    @add_folder('complete')
    def complete(cls, key):
        return cls.config_parser.get('COMPLETE', key)
