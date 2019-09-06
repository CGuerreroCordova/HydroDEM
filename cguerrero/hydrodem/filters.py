"""
Provide the abstract class for filters
"""
from abc import ABC, abstractmethod
import copy
from collections import Counter
import numpy as np
from hydrodem.sliding_window import (SlidingWindow, CircularWindow,
                                     NoCenterWindow, IgnoreBorderInnerSliding)
from scipy.ndimage import binary_erosion


class Filter(ABC):
    """
    Representd a filter
    """

    @abstractmethod
    def apply(self, *args):
        """
        Apply the filter
        Raises
        ------
        NotImplementedError
            If the method was not implemented in the subclass
        """
        raise NotImplementedError


class ComposedFilter(Filter):

    def __init__(self):
        self.filters = []

    def apply(self, image_to_filter):
        content = image_to_filter
        for filter in self.filters:
            content = filter.apply(content)
        return content

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


class QuadraticFilter(Filter):
    """
    Smoothness filter: Apply a quadratic filter of smoothness
    :param dem: dem image
    """

    def __init__(self, *, window_size):
        self.window_size = window_size

    def apply(self, dem):
        values = np.linspace(-self.window_size / 2 + 1, self.window_size / 2,
                             self.window_size)
        xx, yy = np.meshgrid(values, values)
        r0 = self.window_size ** 2
        r1 = (xx * xx).sum()
        r2 = (xx * xx * xx * xx).sum()
        r3 = (xx * xx * yy * yy).sum()

        dem_sliding = SlidingWindow(dem, window_size=self.window_size)
        smoothed = dem.copy()

        for window, center in dem_sliding:
            s1 = window.sum()
            s2 = (window * xx * xx).sum()
            s3 = (window * yy * yy).sum()
            smoothed[center] = ((s2 + s3) * r1 - s1 * (r2 + r3)) / \
                               (2 * r1 ** 2 - r0 * (r2 + r3))
        return smoothed


class CorrectNANValues(Filter):

    def apply(self, dem):
        """
        Correct values lower than zero, generally with extremely lowest values.
        """
        mask_nan = MaskNegatives().apply(dem)
        sliding_nans = SlidingWindow(mask_nan, window_size=3,
                                     iter_over_ones=True)
        dem_sliding = NoCenterWindow(dem, window_size=3)
        for _, center in sliding_nans:
            neighbours_of_nan = dem_sliding[center].ravel()
            neighbours_positives = \
                list(filter(lambda x: x >= 0, neighbours_of_nan))
            dem[center] = sum(neighbours_positives) / len(neighbours_positives)
        return dem


class NegativesValues(Filter):

    def apply(self, image_to_filter):
        # TODO: Condition about values, Exceptions
        return image_to_filter < 0.0


class PositivesValues(Filter):

    def apply(self, image_to_filter):
        # TODO: Condition about values, Exceptions
        return image_to_filter > 0.0


class BooleanToInteger(Filter):

    def apply(self, image_to_filter):
        # TODO: conditions about type of values
        return image_to_filter * 1


class MaskNegatives(ComposedFilter):

    def __init__(self):
        # TODO check to put in properties
        self.filters = [NegativesValues(), BooleanToInteger()]


class MaskPositives(ComposedFilter):

    def __init__(self):
        self.filters = [PositivesValues(), BooleanToInteger()]


class IsolatedPoints(Filter):

    def __init__(self, *, window_size):
        self.window_size = window_size

    def apply(self, image_to_filter):
        """
        Remove isolated pixels detected to be part of a mask.
        """
        sliding = NoCenterWindow(image_to_filter, window_size=self.window_size,
                                 iter_over_ones=True)
        for window, center in sliding:
            image_to_filter[center] = 1. if np.any(window > 0) else 0.
        return image_to_filter


class BlanksFourier(Filter):

    def __init__(self, *, window_size):
        self.window_size = window_size

    def apply(self, image_to_filter):
        """
        Define the filter to detect blanks in a fourier transform image.
        """
        filtered_image = np.zeros(image_to_filter.shape)
        fourier_transform = \
            IgnoreBorderInnerSliding(image_to_filter,
                                     window_size=self.window_size,
                                     inner_size=5)
        for window, center in fourier_transform:
            mean_neighbor = np.nanmean(window)
            real_center = center[0] - self.window_size // 2, \
                          center[1] - self.window_size // 2,
            if image_to_filter[real_center] > (4 * mean_neighbor):
                filtered_image[real_center] = 1.0
        image_modified = image_to_filter * (1 - filtered_image)
        return filtered_image, image_modified


class DetectBlanksFourier(Filter):

    def apply(self, quarter_fourier):
        final_mask_image = np.zeros(quarter_fourier.shape)
        blanks_fourier = BlanksFourier(window_size=55)
        for i in (0, 1):
            filtered_blanks, quarter_fourier = \
                blanks_fourier.apply(quarter_fourier)
            final_mask_image += filtered_blanks
        return final_mask_image


class MaskFourier(ComposedFilter):
    """
    Perform iterations of filter blanks functions and produce a final mask
    with blanks of fourier transform.
    :param quarter_fourier: fourier transform image.
    :return: mask with detected blanks
    """

    def __init__(self):
        # TODO check to put in properties
        self.filters = [DetectBlanksFourier(), IsolatedPoints(window_size=3),
                        ExpandFilter(window_size=13)]
