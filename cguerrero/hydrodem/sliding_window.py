import numpy as np
from itertools import product
from exceptions import (WindowSizeHighError, WindowSizeEvenError,
                        CenterCloseBorderError, NumpyArrayExpectedError,
                        InnerSizeError)

class SlidingWindow:
    """
    Provide an interator of sliding window over a two dimensional ndarray.

    Attributes
    ----------
    grid : ndarray
        The grid on which the iterator will be created to get the sliding
        window
    window_size: int
        The size of the sliding window
    _indices_nan: list(tuple(int, int))
        List of pair of indices of sliding windows which will be set as nan
    iter_over_ones: bool
        Indicates if the iteration will be done only on the grid elements with
        value 1

    Methods
    -------
    customize
        Allow to customize the sliding window returned defining indices to set
        as np.nan. For this class no one element in the window will be set as
        nan.

    Notes
    -----
    The user must know that the values of the sliding window can have nan
    values (due to customising), so it is his duty to take them into account
    when using this subclass.
    The values in the grid will be converted to

    Examples
    --------
    Using iter and next (not usual)
    >>> import numpy as np
    >>> from sliding_window import SlidingWindow
    >>> grid = np.arange(25).reshape((5, 5))
    >>> grid
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])
    >>> sliding = SlidingWindow(grid, window_size=3)
    >>> i = iter(sliding)
    >>> next(i)
    (array([[ 0.,  1.,  2.],
           [ 5.,  6.,  7.],
           [10., 11., 12.]], dtype=float32), (1, 1))
    >>> next(i)
    (array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.],
           [11., 12., 13.]], dtype=float32), (1, 2))
    >>> next(i)
    (array([[ 2.,  3.,  4.],
           [ 7.,  8.,  9.],
           [12., 13., 14.]], dtype=float32), (1, 3))
    >>> next(i)
    (array([[ 5.,  6.,  7.],
           [10., 11., 12.],
           [15., 16., 17.]], dtype=float32), (2, 1))

    Using "for, in"
    >>> grid_2 = np.arange(16).reshape((4, 4))
    >>> grid_2
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> sliding_2 = SlidingWindow(grid_2, window_size=3)
    >>> for window in sliding_2:
    ...     print(window)
    (array([[ 0.,  1.,  2.],
           [ 4.,  5.,  6.],
           [ 8.,  9., 10.]], dtype=float32), (1, 1))
    (array([[ 1.,  2.,  3.],
           [ 5.,  6.,  7.],
           [ 9., 10., 11.]], dtype=float32), (1, 2))
    (array([[ 4.,  5.,  6.],
           [ 8.,  9., 10.],
           [12., 13., 14.]], dtype=float32), (2, 1))
    (array([[ 5.,  6.,  7.],
           [ 9., 10., 11.],
           [13., 14., 15.]], dtype=float32), (2, 2))
    """

    def __init__(self, grid, window_size, iter_over_ones=False):
        """
        Parameters
        ----------
        grid : array_like
            The grid on which the iterator will be created to get the sliding
            window
        window_size: int
            The size of the sliding window
        iter_over_ones: bool, optional
            Indicates if the iteration will be done only on the grid elements
            with value 1 (default is False)
        """
        self.grid = grid
        self.window_size = window_size
        self._indices_nan = []
        self.iter_over_ones = iter_over_ones

    @property
    def grid(self):
        """
        ndarray: Get or set the numpy array for which the iteration is done.

        Raises
        ------
        NumpyArrayExpectedError:
            If the parameters provided is not ndarray type.
        """
        return self._grid

    @grid.setter
    def grid(self, value):
        if not isinstance(value, np.ndarray):
            raise NumpyArrayExpectedError(value)
        self._grid = value.astype('float32')

    @property
    def window_size(self):
        """
        int: Sliding window size. Dimensions of sliding windows will be:
        window_size x window_size

        Raises
        ------
        WindowSizeHighError:
            If the value provided is greater than any of the dimensions of
            the window.
        WindowSizeEvenError:
            If the provided values is not odd.
        """
        return self._window_size

    @window_size.setter
    def window_size(self, value):
        if any(value > i for i in self.grid.shape):
            raise WindowSizeHighError(value, self.grid.shape)
        elif value % 2 != 1:
            raise WindowSizeEvenError(value)
        self._window_size = int(value)

    def __iter__(self):
        """
        Define the iterator for the SlidingWindow class. Use window size and
        grid dimensions to create elements to produce. Allow customize the
        window produces in terms of set some values of the window as nan, this
        way differents kind of windoes can be customized and created. For one
        instance only one type of customization is allowed.

        Yields
        -------
        ndarray, tuple(int, int)
            * The next sliding window, going through the grid by row and by
            column, that is, for each row it moves from the left to the right
            until the end of the row and then go to the next row, continuing
            to the last element of the last row.
            * A tuple with actual indices of iteration in relation with input
            grid

        Notes
        -----
        Are presented two examples. One of them using 'iter' and 'next', and
        the other one using 'for, in'
        """
        ny, nx = self.grid.shape
        left_up = self.window_size // 2
        right_down = left_up + 1

        self._customize()

        def range_grid(bound_index):
            return range(left_up, (bound_index - right_down) + 1)

        for j in range_grid(ny):
            for i in range_grid(nx):
                if not self.iter_over_ones or int(self.grid[j, i]) == 1:
                    neighbour = self.grid[j - left_up: j + right_down,
                                i - left_up: i + right_down]
                    neighbour = self._set_nan(neighbour)
                    yield neighbour, (j, i)

    def __getitem__(self, coords):
        """
        Slice a window with dimensions of window_size * window_size attribute
        class from grid attribute class, centered in indices indicated by
        coords. If some class inherits from this class and customized method is
        override, some elements of the window can be nan, depending of the
        customization

        Parameters
        ----------
        coords: tuple(int, int)
            Coordinates to get the window from grid

        Returns
        -------
        ndarray
            The window sliced from grid

        Raises
        ------
        CenterCloseBorderError
            If coordinates of the center to slice is too close to the border

        Examples
        --------
        >>> import numpy as np
        >>> from sliding_window import SlidingWindow
        >>> grid = np.arange(25).reshape((5, 5))
        >>> grid
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24]])
        >>> sliding = SlidingWindow(grid, window_size=3)
        >>> sliding[1, 1]
        array([[ 0.,  1.,  2.],
               [ 5.,  6.,  7.],
               [10., 11., 12.]], dtype=float32)
        >>> sliding[3,3]
        array([[12., 13., 14.],
               [17., 18., 19.],
               [22., 23., 24.]], dtype=float32)
        >>> sliding[1,3]
        array([[ 2.,  3.,  4.],
               [ 7.,  8.,  9.],
               [12., 13., 14.]], dtype=float32)
        """
        j, i = coords
        left_up = self.window_size // 2
        right_down = left_up + 1

        self._customize()

        def _get_slice(coord):
            """
            Return slice corresponding to coord minus the middle of the window
            size
            """
            return slice(coord - left_up, coord + right_down)

        if self._check_border(j, i, left_up, right_down):
            neighbours = self.grid[_get_slice(j), _get_slice(i)]
            neighbours = self._set_nan(neighbours)
            return neighbours
        else:
            raise CenterCloseBorderError(coords, window_size=self.window_size)

    def _check_border(self, j, i, left_up, right_down):
        """
        Check if indices j, i are too close to the border
        """
        ny, nx = self.grid.shape
        return all(index >= left_up for index in (j, i)) and \
               j <= (ny - right_down) and \
               i <= (nx - right_down)

    def _customize(self):
        """
        Allow to customize the sliding window returned defining indices to set
        as np.nan. For this class no one element in the window will be set as
        nan.
        """
        pass

    def _set_nan(self, neighbours):
        """
        Set as nan elements of the window. The elements defined
        in self._indices_nan will be set as nan, the rest of elements keep its
        value. The window modified and returned is copied in this method.

        Parameters
        ----------
        neighbours: ndarray
            window to set some element as nan.

        Returns
        -------
        ndarray:
            A copy of the input window with some or none elements set as nan

        """
        cp_window = neighbours.copy()
        for indices in self._indices_nan:
            cp_window[indices] = np.nan
        return cp_window


class SlidingIgnoreBorder(SlidingWindow):
    """
    Modify the functionality of SlidingWindow iterator changing the unlerlying
    grid by adding an extra margin to grid attribute, this way it allows to have
    sliding window that ignores original border of the grid. The new margin
    will be set with nan values, so it doesn't modify the values of the grid.

    Examples
    --------
    Using iter and next (not usual)
    >>> import numpy as np
    >>> from sliding_window import SlidingIgnoreBorder
    >>> grid = np.arange(25).reshape((5, 5))
    >>> grid
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])
    >>> sliding = SlidingIgnoreBorder(grid, window_size=3)
    >>> i = iter(sliding)
    >>> next(i)
    (array([[nan, nan, nan],
           [nan,  0.,  1.],
           [nan,  5.,  6.]], dtype=float32), (1, 1))
    >>> next(i)
    (array([[nan, nan, nan],
           [ 0.,  1.,  2.],
           [ 5.,  6.,  7.]], dtype=float32), (1, 2))
    >>> next(i)
    (array([[nan, nan, nan],
           [ 1.,  2.,  3.],
           [ 6.,  7.,  8.]], dtype=float32), (1, 3))
    >>> next(i)
    (array([[nan, nan, nan],
           [ 2.,  3.,  4.],
           [ 7.,  8.,  9.]], dtype=float32), (1, 4))

    Example using "for, in"
    >>> grid_2 = np.arange(9).reshape((3, 3))
    >>> grid_2
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> sliding_2 = SlidingIgnoreBorder(grid_2, window_size=3)
    >>> for window in sliding_2:
    ...     print(window)
    (array([[nan, nan, nan],
           [nan,  0.,  1.],
           [nan,  3.,  4.]], dtype=float32), (1, 1))
    (array([[nan, nan, nan],
           [ 0.,  1.,  2.],
           [ 3.,  4.,  5.]], dtype=float32), (1, 2))
    (array([[nan, nan, nan],
           [ 1.,  2., nan],
           [ 4.,  5., nan]], dtype=float32), (1, 3))
    (array([[nan,  0.,  1.],
           [nan,  3.,  4.],
           [nan,  6.,  7.]], dtype=float32), (2, 1))
    (array([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]], dtype=float32), (2, 2))
    (array([[ 1.,  2., nan],
           [ 4.,  5., nan],
           [ 7.,  8., nan]], dtype=float32), (2, 3))
    (array([[nan,  3.,  4.],
           [nan,  6.,  7.],
           [nan, nan, nan]], dtype=float32), (3, 1))
    (array([[ 3.,  4.,  5.],
           [ 6.,  7.,  8.],
           [nan, nan, nan]], dtype=float32), (3, 2))
    (array([[ 4.,  5., nan],
           [ 7.,  8., nan],
           [nan, nan, nan]], dtype=float32), (3, 3))
    """

    def __init__(self, grid, window_size, *args, **kwargs):
        """
        Extends the parent constructor modifying the grid with the addition of
        an extra margin to the grid of size: window_size

        Parameters
        ----------
        grid : ndarray
            The grid on which the iterator will be created to get the sliding
            window
        window_size: int
            The size of the sliding window
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(grid, window_size, *args, **kwargs)
        self.grid = self.__add_extra_margin()

    def __add_extra_margin(self):
        """
        Add an extra margin to all sides of self.grid of size self.window_size.
        To do this create a new grid greater than self.grid (filled with nan
        values) and copy the content of self.grid centered in the new grid

        Returns
        -------
        ndarray
            The new grid with an extra margin added.
        """
        ny, nx = self.grid.shape
        ny = ny + self.window_size - 1
        nx = nx + self.window_size - 1
        middle = self.window_size // 2
        grid_expanded = np.empty((ny, nx))
        grid_expanded[:] = np.nan
        grid_expanded[middle:ny - middle, middle:nx - middle] = \
            self.grid.astype('float32')
        return grid_expanded

class CircularWindow(SlidingWindow):
    """
    Modify the functionality of the SlidingWindow iterator extending the
    customization to include a circular window, that is adding corners
    element of window to be set as nan.

    Examples
    --------
    Using iter and next (not usual)
    >>> import numpy as np
    >>> from sliding_window import CircularWindow
    >>> grid = np.arange(25).reshape((5, 5))
    >>> grid
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])
    >>> sliding = CircularWindow(grid, window_size=3)
    >>> i = iter(sliding)
    >>> next(i)
    (array([[nan,  1., nan],
           [ 5.,  6.,  7.],
           [nan, 11., nan]], dtype=float32), (1, 1))
    >>> next(i)
    (array([[nan,  2., nan],
           [ 6.,  7.,  8.],
           [nan, 12., nan]], dtype=float32), (1, 2))
    >>> next(i)
    (array([[nan,  3., nan],
           [ 7.,  8.,  9.],
           [nan, 13., nan]], dtype=float32), (1, 3))
    >>> next(i)
    (array([[nan,  6., nan],
           [10., 11., 12.],
           [nan, 16., nan]], dtype=float32), (2, 1))

    Example using "for, in"
    >>> grid_2 = np.arange(12).reshape((3, 4))
    >>> grid_2
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    >>> sliding_2 = CircularWindow(grid_2, window_size=3)
    >>> for window in sliding_2:
    ...     print(window)
    (array([[nan,  1., nan],
           [ 4.,  5.,  6.],
           [nan,  9., nan]], dtype=float32), (1, 1))
    (array([[nan,  2., nan],
           [ 5.,  6.,  7.],
           [nan, 10., nan]], dtype=float32), (1, 2))
    """

    def _customize(self):
        """
        Extends the customize method to include in self._indices_nan
        elements in the corners of the sliding window
        Also it calls super customize method so, other customization can be
        done for any cooperative class.
        """
        self._indices_nan.extend(self._remove_corners(self.window_size))
        super()._customize()

    def _remove_corners(self, ny):
        """
        Define corners indices pairs using window size

        Parameters
        ----------
        ny :  int
            window size

        Returns
        -------
        list(tuple(int, int))
            List of indice pairs corresponding to window corners.
        """
        return [(0, 0), (0, ny - 1), (ny - 1, 0), (ny - 1, ny - 1)]


class InnerWindow(SlidingWindow):
    """
    Modify the functionality of the SlidingWindow iterator extending the
    customization to include a inner window inside the window returned in each
    iteration. The inner window is composed by nan values. The inner window
    doesn't include the center pixel unless the size of inner size is 1.

    Attributes
    ----------
    inner_size : array_like
        Size of inner window inside the sliding window of iteration

    Examples
    --------
    Using iter and next (not usual)
    >>> import numpy as np
    >>> from sliding_window import InnerWindow
    >>> grid = np.arange(81).reshape((9, 9))
    >>> grid
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
           [ 9, 10, 11, 12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23, 24, 25, 26],
           [27, 28, 29, 30, 31, 32, 33, 34, 35],
           [36, 37, 38, 39, 40, 41, 42, 43, 44],
           [45, 46, 47, 48, 49, 50, 51, 52, 53],
           [54, 55, 56, 57, 58, 59, 60, 61, 62],
           [63, 64, 65, 66, 67, 68, 69, 70, 71],
           [72, 73, 74, 75, 76, 77, 78, 79, 80]])
    >>> sliding = InnerWindow(grid, window_size=5, inner_size=3)
    >>> i = iter(sliding)
    >>> next(i)
    (array([[ 0.,  1.,  2.,  3.,  4.],
           [ 9., nan, nan, nan, 13.],
           [18., nan, 20., nan, 22.],
           [27., nan, nan, nan, 31.],
           [36., 37., 38., 39., 40.]], dtype=float32), (2, 2))
    >>> next(i)
    (array([[ 1.,  2.,  3.,  4.,  5.],
           [10., nan, nan, nan, 14.],
           [19., nan, 21., nan, 23.],
           [28., nan, nan, nan, 32.],
           [37., 38., 39., 40., 41.]], dtype=float32), (2, 3))
    >>> next(i)
    (array([[ 2.,  3.,  4.,  5.,  6.],
           [11., nan, nan, nan, 15.],
           [20., nan, 22., nan, 24.],
           [29., nan, nan, nan, 33.],
           [38., 39., 40., 41., 42.]], dtype=float32), (2, 4))
    >>> next(i)
    (array([[ 3.,  4.,  5.,  6.,  7.],
           [12., nan, nan, nan, 16.],
           [21., nan, 23., nan, 25.],
           [30., nan, nan, nan, 34.],
           [39., 40., 41., 42., 43.]], dtype=float32), (2, 5))

    Example using "for, in"
    >>> grid_2 = np.arange(36).reshape((6, 6))
    >>> grid_2
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34, 35]])
    >>> sliding_2 = InnerWindow(grid_2, window_size=5, inner_size=3)
    >>> for window in sliding_2:
    ...     print(window)
    (array([[ 0.,  1.,  2.,  3.,  4.],
           [ 6., nan, nan, nan, 10.],
           [12., nan, 14., nan, 16.],
           [18., nan, nan, nan, 22.],
           [24., 25., 26., 27., 28.]], dtype=float32), (2, 2))
    (array([[ 1.,  2.,  3.,  4.,  5.],
           [ 7., nan, nan, nan, 11.],
           [13., nan, 15., nan, 17.],
           [19., nan, nan, nan, 23.],
           [25., 26., 27., 28., 29.]], dtype=float32), (2, 3))
    (array([[ 6.,  7.,  8.,  9., 10.],
           [12., nan, nan, nan, 16.],
           [18., nan, 20., nan, 22.],
           [24., nan, nan, nan, 28.],
           [30., 31., 32., 33., 34.]], dtype=float32), (3, 2))
    (array([[ 7.,  8.,  9., 10., 11.],
           [13., nan, nan, nan, 17.],
           [19., nan, 21., nan, 23.],
           [25., nan, nan, nan, 29.],
           [31., 32., 33., 34., 35.]], dtype=float32), (3, 3))
    """

    def __init__(self, grid, window_size, inner_size, *args, **kwargs):
        """
        Extends the parent constructor adding a new attribute inner_size that
        keep the value of size of inner window

        Parameters
        ----------
        grid : array_like
            The grid on which the iterator will be created to get the sliding
            window
        window_size : int
            The size of the sliding window
        inner_size : int
            The size of the inner window inside the sliding window
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        self.inner_size = inner_size
        super().__init__(grid, window_size, *args, **kwargs)

    def _customize(self):
        """
        Extends the customize method to include into self._indices_nan
        elements of the inner window.
        Also it calls super customize method so, others customization can be
        done for any cooperative class.
        """
        self._indices_nan.extend(self._inner_window(self.window_size,
                                                    self.inner_size))
        super()._customize()

    def _inner_window(self, ny, inner_size):
        """
        Define inner window indices pairs using inner_size

        Parameters
        ----------
        ny :  int
            window size
        inner_size : int
            inner window size

        Raises
        ------
        InnerSizeError
            If inner_size is greater than window_size attribute

        Returns
        -------
        list(tuple(int, int))
            List of indices pairs corresponding to inner window.
        """

        if inner_size > self.window_size:
            raise InnerSizeError(inner_size, self.window_size)
        else:
            ratio_inner = inner_size // 2
            center = ny // 2
            indices = (center - ratio_inner, center + ratio_inner + 1)
            pairs = product(range(*indices), range(*indices))
            # Removing the center from the list to set nan
            inner_window = ((x, y) for x, y in pairs if not x == y == center)
            return inner_window


class NoCenterWindow(SlidingWindow):
    """
    Modify the functionality of the SlidingWindow iterator extending the
    customization to omit the center of the window returned in each
    iteration.

    Examples
    --------
    Using iter and next (not usual)
    >>> import numpy as np
    >>> from sliding_window import NoCenterWindow
    >>> grid = np.arange(25).reshape((5, 5))
    >>> grid
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])
    >>> sliding = NoCenterWindow(grid, window_size=3)
    >>> i = iter(sliding)
    >>> next(i)
    (array([[ 0.,  1.,  2.],
           [ 5., nan,  7.],
           [10., 11., 12.]], dtype=float32), (1, 1))
    >>> next(i)
    (array([[ 1.,  2.,  3.],
           [ 6., nan,  8.],
           [11., 12., 13.]], dtype=float32), (1, 2))
    >>> next(i)
    (array([[ 2.,  3.,  4.],
           [ 7., nan,  9.],
           [12., 13., 14.]], dtype=float32), (1, 3))
    >>> next(i)
    (array([[ 5.,  6.,  7.],
           [10., nan, 12.],
       [15., 16., 17.]], dtype=float32), (2, 1))

    Example using "for, in"
    >>> grid_2 = np.arange(12).reshape((3, 4))
    >>> grid_2
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    >>> sliding_2 = NoCenterWindow(grid_2, window_size=3)
    >>> for window in sliding_2:
    ...     print(window)
    (array([[ 0.,  1.,  2.],
           [ 4., nan,  6.],
           [ 8.,  9., 10.]], dtype=float32), (1, 1))
    (array([[ 1.,  2.,  3.],
           [ 5., nan,  7.],
           [ 9., 10., 11.]], dtype=float32), (1, 2))
    """

    def _customize(self):
        """
        Extends the customize method to include into self._indices_nan
        elements the center element of the sliding window.
        Also it calls super customize method so, others customization can be
        done for any cooperative class.
        """
        self._indices_nan.extend(self._center_index(self.window_size))
        super()._customize()

    def _center_index(self, ny):
        """
        Compute the center of the window

        Parameters
        ----------
        ny :  int
            window size

        Returns
        -------
        list(tuple(int, int))
            List containing a pair with the center indices of the sliding
            window
        """
        center = ny // 2
        return [(center, center)]


class IgnoreBorderInnerSliding(SlidingIgnoreBorder, InnerWindow,
                               NoCenterWindow):
    """
    Represent an Sliding Window iterator that:
        Ignore Borders
        Have an Inner Window
        The Center element is omitted
    This is a cooperative inheritance using the customising of elements to be
    set as nan. Constructor and methods are omitted because are in charge of
    the parent classes combination.

    Examples
    --------
    Using iter and next (not usual)
    >>> import numpy as np
    >>> from sliding_window import IgnoreBorderInnerSliding
    >>> grid = np.arange(81).reshape((9, 9))
    >>> grid
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
           [ 9, 10, 11, 12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23, 24, 25, 26],
           [27, 28, 29, 30, 31, 32, 33, 34, 35],
           [36, 37, 38, 39, 40, 41, 42, 43, 44],
           [45, 46, 47, 48, 49, 50, 51, 52, 53],
           [54, 55, 56, 57, 58, 59, 60, 61, 62],
           [63, 64, 65, 66, 67, 68, 69, 70, 71],
           [72, 73, 74, 75, 76, 77, 78, 79, 80]])
    >>> sliding = IgnoreBorderInnerSliding(grid, window_size=5, inner_size=3)
    >>> i = iter(sliding)
    >>> next(i)
    (array([[nan, nan, nan, nan, nan],
           [nan, nan, nan, nan, nan],
           [nan, nan, nan, nan,  2.],
           [nan, nan, nan, nan, 11.],
           [nan, nan, 18., 19., 20.]], dtype=float32), (2, 2))
    >>> next(i)
    (array([[nan, nan, nan, nan, nan],
           [nan, nan, nan, nan, nan],
           [nan, nan, nan, nan,  3.],
           [nan, nan, nan, nan, 12.],
           [nan, 18., 19., 20., 21.]], dtype=float32), (2, 3))
    >>> next(i)
    (array([[nan, nan, nan, nan, nan],
           [nan, nan, nan, nan, nan],
           [ 0., nan, nan, nan,  4.],
           [ 9., nan, nan, nan, 13.],
           [18., 19., 20., 21., 22.]], dtype=float32), (2, 4))
    >>> next(i)
    (array([[nan, nan, nan, nan, nan],
           [nan, nan, nan, nan, nan],
           [ 1., nan, nan, nan,  5.],
           [10., nan, nan, nan, 14.],
           [19., 20., 21., 22., 23.]], dtype=float32), (2, 5))

    """
    pass
