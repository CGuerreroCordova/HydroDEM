import numpy as np
from itertools import product
from .exceptions import (WindowSizeHighError, WindowSizeEvenError,
                         CenterCloseBorderError, NumpyArrayExpectedError)

class SlidingWindow:

    def __init__(self, grid, window_size, iter_over_ones=False, *args,
                 **kwargs):
        self.grid = grid
        self.window_size = window_size
        self._indices_nan = []
        self.grid_nan = np.nan
        self.iter_over_ones = iter_over_ones

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        if not isinstance(value, np.ndarray):
            raise NumpyArrayExpectedError(value)
        self._grid = value.astype('float32')

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, value):
        if value > self.grid.shape[0] or value > self.grid.shape[1]:
            raise WindowSizeHighError(value, self.grid.shape)
        elif value % 2 != 1:
            raise WindowSizeEvenError(value)
        self._window_size = int(value)

    def __iter__(self):
        ny, nx = self.grid.shape
        left_up = self.window_size // 2
        right_down = left_up + 1

        self.customize()

        def range_grid(bound_index):
            return range(left_up, (bound_index - right_down) + 1)

        for j in range_grid(ny):
            for i in range_grid(nx):
                if not self.iter_over_ones or int(self.grid[j, i]) == 1:
                    neighbour = self.grid[j - left_up: j + right_down,
                                i - left_up: i + right_down]
                    neighbour = self.set_nan(neighbour)
                    yield neighbour, (j, i)

    def __getitem__(self, coords):
        j, i = coords
        left_up = self.window_size // 2
        right_down = left_up + 1

        self.customize()
        if self._check_border(j, i, left_up, right_down):
            neighbours = self.grid[j - left_up: j + right_down,
                         i - left_up: i + right_down]
            neighbours = self.set_nan(neighbours)
            return neighbours
        else:
            raise CenterCloseBorderError(coords, window_size=self.window_size)

    def _check_border(self, j, i, left_up, right_down):
        ny, nx = self.grid.shape
        return all(index >= left_up for index in (j, i)) and \
               j <= (ny - right_down) + 1 and \
               i <= (nx - right_down) + 1

    def customize(self):
        pass

    def set_nan(self, neighbours):
        cp_window = neighbours.copy()
        for index in self._indices_nan:
            cp_window[index] = self.grid_nan
        return cp_window


class SlidingIgnoreBorder(SlidingWindow):

    def __init__(self, grid, window_size, *args, **kwargs):
        super().__init__(grid, window_size, *args, **kwargs)
        self.grid = self._add_extra_margin()

    def _add_extra_margin(self):
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

    def __init__(self, grid, window_size, *args, **kwargs):
        super().__init__(grid, window_size, *args, **kwargs)

    def customize(self):
        self._indices_nan.extend(self.remove_corners(self.window_size))
        super().customize()

    def remove_corners(self, ny):
        return [(0, 0), (0, ny - 1), (ny - 1, 0), (ny - 1, ny - 1)]


class InnerWindow(SlidingWindow):

    def __init__(self, grid, window_size, inner_size, *args, **kwargs):
        self.inner_size = inner_size
        super().__init__(grid, window_size, *args, **kwargs)

    def customize(self):
        self._indices_nan.extend(self.inner_window(self.window_size,
                                                   self.inner_size))
        super().customize()

    def inner_window(self, ny, inner_size):
        # TODO, conditions
        ratio_inner = inner_size // 2
        center = ny // 2
        indices = (center - ratio_inner, center + ratio_inner + 1)
        pairs = product(range(*indices), range(*indices))
        # Removing the center from the list to set nan
        inner_window = ((x, y) for x, y in pairs if not x == y == center)
        return inner_window


class NoCenterWindow(SlidingWindow):

    def __init__(self, grid, window_size, *args, **kwargs):
        super().__init__(grid, window_size, *args, **kwargs)

    def customize(self):
        self._indices_nan.extend(self.center_index(self.window_size))
        super().customize()

    def center_index(self, ny):
        center = ny // 2
        return [(center, center)]


class CombineWindows(NoCenterWindow, InnerWindow, CircularWindow):

    def __init__(self, grid, *, window_size, inner_size):
        super().__init__(grid=grid, window_size=window_size,
                         inner_size=inner_size)

    def customize(self):
        super().customize()


class IgnoreBorderInnerSliding(SlidingIgnoreBorder, InnerWindow,
                               NoCenterWindow):
    def __init__(self, grid, *, window_size, inner_size):
        super().__init__(grid=grid, window_size=window_size,
                         inner_size=inner_size)
