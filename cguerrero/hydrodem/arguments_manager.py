import argparse
import sys


class ArgumentsManager(object):
    """
    Responsible to parse the input arguments from bash
    """

    def __init__(self):
        """
        Default constructor
        :param cmd_line: the command line
        :type cmd_line: str
        """
        self._parser = argparse.ArgumentParser(description="Hydro DEM",
                                               epilog="Use %(prog)s {command} "
                                                      "-h to get help")
        self._setup_arguments()

    def _setup_arguments(self):
        """
        Sets the input arguments
        """

        self._parser.add_argument("-a", "--area-interest",
                                  help="Area of interest to process, "
                                       "shapefile path", required=True)
        # FUTURE VERSIONS
        # self._parser.add_argument("-s", "--srtm-dem",
        #                           help="Path to SRTM DEM file. Zip format",
        #                           required=False)
        # self._parser.add_argument("-y", "--hsheds-dem",
        #                           help="Path to HSHEDS DEM file. Zip format",
        #                           required=False)
        # self._parser.add_argument("-g", "--groves-file",
        #                           help="Path to groves classification file. "
        #                                "Zip format",
        #                           required=False)

    def parse(self, command_line=sys.argv[1:]):
        """
        Parses the command line arguments
        :return parameters parsed
        :rtype: argparse
        """
        return self._parser.parse_args(command_line)
