import numpy as np


class SlidingWindow:
    indices_nan = []

    def __init__(self, grid, window_size):
        self.grid = grid
        self.window_size = window_size

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

        def range_grid(max_index):
            return range(left_up, (max_index - right_down) + 1)

        for j in range_grid(ny):
            for i in range_grid(nx):
                neighbors = self.grid[j - left_up: j + right_down,
                            i - left_up: i + right_down]
                neighbors = self.customize(neighbors)
                yield neighbors

    def customize(self, neighbors):
        return neighbors

    def set_nan(self, array, indices):
        for index in indices:
            array[index] = np.nan


class CircularWindow(SlidingWindow):

    def customize(self, neighbors):
        ny, nx = neighbors.shape
        indices_nan = self.remove_corners(ny, nx)
        cp_window = neighbors.copy()
        self.set_nan(cp_window, indices_nan)
        return (cp_window)

    def remove_corners(self, ny, nx):
        return [(0, 0), (0, nx - 1), (ny - 1, 0), (ny - 1, nx - 1)]
