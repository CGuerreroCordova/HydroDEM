from configparser import ConfigParser
import os


class Config:
    """Interact with configuration variables."""

    BASEDIR = os.getcwd()
    config_parser = ConfigParser()
    config_parser.optionxform = str
    config_ini = (os.path.join(os.getcwd(), 'config.ini'))

    @classmethod
    def initialize(cls, config_file=None):
        """Start config by reading config.ini."""
        if config_file:
            config = config_file
        else:
            config = cls.config_ini
        cls.config_parser.read(config)

    @classmethod
    def srtm(cls, key):
        """Get srtm values from config.ini."""
        return cls.config_parser.get('SRTM', key)

    @classmethod
    def hsheds(cls, key):
        """Get hsheds values from config.ini."""
        return cls.config_parser.get('HSHEDS', key)

    @classmethod
    def rivers(cls, key):
        """Get rivers values from config.ini."""
        return cls.config_parser.get('RIVERS', key)

    @classmethod
    def shapes(cls, key):
        """Get shapes values from config.ini."""
        return cls.config_parser.get('SHAPES', key)

    @classmethod
    def groves(cls, key):
        """Get groves values from config.ini."""
        return cls.config_parser.get('GROVES', key)

    @classmethod
    def final(cls, key):
        """Get final values from config.ini."""
        return cls.config_parser.get('FINAL', key)

    @classmethod
    def temp(cls, key):
        """Get temp values from config.ini."""
        return cls.config_parser.get('TEMP', key)

    @classmethod
    def profiles(cls, key):
        """Get profiles values from config.ini."""
        return cls.config_parser.get('PROFILES', key)
