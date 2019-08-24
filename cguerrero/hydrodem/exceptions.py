"""
Provide some exceptions to use in HydroDEM
"""


class HydroDEMException(Exception):
    """
    Represents the base exception for HydroDEM
    """

    def __init__(self, msg=""):
        """
        Default constructor
        """
        self._msg = msg
        super().__init__(self)

    def __str__(self):
        """
        Returns the exception message
        """
        return self._msg


class WindowSizeHighError(HydroDEMException):
    """
    Raises when a window size is higher than grid dimension
    """

    def __init__(self, window_size, grid_dimensions=""):
        super().__init__(msg=f'Window size: {window_size} cannot be higher'
                             f' than grid dimensions: {grid_dimensions}')


class WindowSizeEvenError(HydroDEMException):
    """
    Raises when an even window size is provided
    """

    def __init__(self, window_size):
        super().__init__(msg=f'Window size: {window_size} cannot be an even '
                             f'number')


class CenterCloseBorderError(HydroDEMException):
    """
    Raise when a window is requested and the center of window is close to the
    border, then, the window cannot be get.
    """

    def __init__(self, center_window, window_size):
        super().__init__(msg=f'Center of window: {center_window} too close of '
                             f'border. Window size: {window_size}')


class NumpyArrayExpectedError(HydroDEMException):
    """
    Raise when a non numpy.ndarray element is provided
    """

    def __init__(self, provided):
        super().__init__(msg=f'Expected numpy ndarray type. Provided: '
                             f'{type(provided)}')


class InnerSizeError(HydroDEMException):
    """
    Raise when a non numpy.ndarray element is provided
    """

    def __init__(self, provided):
        super().__init__(msg=f'Expected numpy ndarray type. Provided: '
                             f'{type(provided)}')
