"""
Provide the abstract class for filters
"""
from abc import ABC, abstractmethod
import copy
from collections import Counter
import numpy as np
from hydrodem.sliding_window import (SlidingWindow, CircularWindow)
from scipy.ndimage import binary_erosion


class Filter(ABC):
    """
    Representd a filter
    """

    @abstractmethod
    def apply(self):
        """
        Apply the filter
        Raises
        ------
        NotImplementedError
            If the method was not implemented in the subclass
        """
        raise NotImplementedError


class MajorityFilter(Filter):
    """
    The value assigned to the center will be the most frequent value,
    contained in a windows of window_size x window_size
    """

    def __init__(self, *, window_size):
        self.window_size = window_size

    def apply(self, image_to_filter):
        filtered_image = np.zeros(image_to_filter.shape)
        sliding = CircularWindow(image_to_filter, self.window_size)
        for window, center in sliding:
            frequency = Counter(window.ravel())
            value, count = frequency.most_common()[0]
            if count > (self.window_size ** 2 - 1) * 0.7:
                filtered_image[center] = value
        return filtered_image


class BinaryErosion(Filter):
    """
    Represents the Mmajority Filter
    """

    def __init__(self, *, iterations):
        self.iterations = iterations

    def apply(self, image_to_filter):
        return binary_erosion(image_to_filter, iterations=self.iterations)


class ExpandFilter(Filter):
    """
    The value assigned to the center will be 1 if at least one pixel
    inside the circular window is 1
    """

    def __init__(self, *, window_size):
        self.window_size = window_size

    def apply(self, img_to_expand):
        expanded_image = np.zeros(img_to_expand.shape)
        sliding = CircularWindow(img_to_expand, self.window_size)
        for window, center in sliding:
            if np.any(window > 0):
                expanded_image[center] = 1
        return expanded_image


class EnrouteRivers(Filter):
    """
    Apply homogeneity to canyons. Specific defined for images with flow stream.
    """

    def __init__(self, *, window_size):
        self.window_size = window_size

    def apply(self, dem_in, mask_rivers):
        dem = copy.deepcopy(dem_in)
        left_up = self.window_size // 2
        rivers_enrouted = np.zeros(dem.shape)
        sliding = SlidingWindow(mask_rivers, window_size=self.window_size,
                                iter_over_ones=True)
        dem_sliding = SlidingWindow(dem, window_size=self.window_size)
        for _, (j, i) in sliding:
            window_dem = dem_sliding[j, i]
            neighbour_min = np.amin(window_dem.ravel())
            indices_min = np.nonzero(window_dem == neighbour_min)
            for min_j, min_i in zip(indices_min[0], indices_min[1]):
                indices = (j - left_up + min_j, i - left_up + min_i)
                rivers_enrouted[indices] = 1
                dem_sliding.grid[indices] = 10000
        return rivers_enrouted
