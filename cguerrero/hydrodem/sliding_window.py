import numpy as np
from itertools import product

class SlidingWindow:

    def __init__(self, grid, window_size, *args, **kwargs):
        self.grid = grid
        self.window_size = window_size
        self._indices_nan = []
        self.grid_nan = np.nan

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError("Expected numpy ndarray")
        self._grid = value.astype('float32')

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, value):
        if value > self.grid.shape[0] or value > self.grid.shape[1]:
            raise ValueError("Window size must be lower than grid dimension")
        elif value % 2 != 1:
            raise ValueError("Window size must be odd.")
        self._window_size = int(value)

    def __iter__(self):
        ny, nx = self.grid.shape
        left_up = self.window_size // 2
        right_down = left_up + 1

        self.customize()

        def range_grid(max_index):
            return range(left_up, (max_index - right_down) + 1)

        for j in range_grid(ny):
            for i in range_grid(nx):
                neighbors = self.grid[j - left_up: j + right_down,
                            i - left_up: i + right_down]
                neighbors = self.set_nan(neighbors)
                yield neighbors, (j, i)

    def customize(self):
        pass

    def set_nan(self, neighbors):
        cp_window = neighbors.copy()
        for index in self._indices_nan:
            cp_window[index] = self.grid_nan
        return cp_window


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

    def __init__(self, grid, window_size, inner_size):
        super().__init__(grid=grid, window_size=window_size,
                         inner_size=inner_size)

    def customize(self):
        super().customize()
