import copy
from collections import Counter
import numpy as np
from filters.simple_filters import (LowerThan, BooleanToInteger, GreaterThan,
                                    ProductFilter, SubtractionFilter,
                                    AdditionFilter)
from hydrodem.sliding_window import (SlidingWindow, CircularWindow,
                                     NoCenterWindow, IgnoreBorderInnerSliding)
from filters import Filter, ComposedFilter, ComposedFilterResults
from filters.extension_filters import (BitwiseXOR, BinaryErosion, Around,
                                       BinaryClosing, GreyDilation, Convolve,
                                       FourierITransform, FourierTransform,
                                       FourierShift, FourierIShift,
                                       AbsolutValues)


class MajorityFilter(Filter):
    """
    Use a Circular Sliding Window to go through the image and filter it
    assigning to the center of the window the most frequent value contained in
    the sliding window.

    Attributes
    ----------
    window_size : int
        Window size of circular sliding windows used to compute the filter.

    Methods
    -------
    apply
        apply the filter
    """

    def __init__(self, *, window_size):
        """
        Parameters
        ----------
        window_size : int
            Window size of circular sliding windows used to compute the filter.
        """
        self.window_size = window_size

    def apply(self, image_to_filter):
        """
        Apply the filter. Create a zeros grid to store the filtered image.
        Create a Circular Sliding Window. The frequency taken into account to
        decide if it the most frequency value is if the value is present more
        than 70 percent of elements of the window. Not taking into account the
        center value as an element.

        Parameters
        ----------
        image_to_filter: ndarray
            Image to apply the majority filter

        Returns
        -------
        ndarray
            New object containing the filtered image.
        """
        filtered_image = np.zeros(image_to_filter.shape)
        sliding = CircularWindow(image_to_filter, self.window_size)
        for window, center in sliding:
            frequency = Counter(window.ravel())
            value, count = frequency.most_common()[0]
            if count > (self.window_size ** 2 - 1) * 0.7:
                filtered_image[center] = value
        return filtered_image


class ExpandFilter(Filter):
    """
    Use a Circular Sliding Window to go through the image and filter it
    assigning to the center of the window the value 1 if at least one pixel
    inside the circular window is 1, 0 otherwise.

    Attributes
    ----------
    window_size : int
        Window size of circular sliding windows used to compute the filter.

    Methods
    -------
    apply
        apply the filter
    """

    def __init__(self, *, window_size):
        """
        Parameters
        ----------
        window_size : int
            Window size of circular sliding windows used to compute the filter.
        """
        self.window_size = window_size

    def apply(self, image_to_filter):
        """
        Apply the filter. Create a zeros grid to store the filtered image.
        Create a Circular Sliding Window. Remove nan value coming from the
        sliding window. The value for the center of the window will be 1 if at
        least one pixel inside the circular window is 1, 0 otherwise.

        Parameters
        ----------
        image_to_filter: ndarray
            Image to apply the majority filter

        Returns
        -------
        ndarray
            New object containing the expanded new image.
        """
        expanded_image = np.zeros(image_to_filter.shape)
        sliding = CircularWindow(image_to_filter, self.window_size)
        for window, center in sliding:
            window = window[~np.isnan(window)]
            if np.any(window > 0):
                expanded_image[center] = 1
        return expanded_image


class RouteRivers(Filter):
    """
    Route and/or expand rivers coming from a mask image. Use another dem
    image as reference to decide the routing of rivers. Reference DEM must
    be provided in the constructor, the mask image when the filter is applied.
    Use a sliding window iterating through the mask image with values 1.
    With the center value as reference, check values contained in the window
    of dem of reference to add new candidates to be river from the low values
    inside the window.

    Attributes
    ----------
    window_size : int
        Window size of circular sliding windows used to compute the filter.
    dem : ndarray
        Dem used as a value reference to decide the belonging of elements to
        rivers

    Methods
    -------
    apply
        apply the filter
    """

    def __init__(self, *, window_size, dem):
        """
        Parameters
        ----------
        window_size : int
            Window size of circular sliding windows used to compute the filter.
        dem : ndarray
            Dem used as a value reference to decide the belonging of elements
            to rivers
        """
        self.window_size = window_size
        self.dem = copy.deepcopy(dem)

    def apply(self, mask_rivers):
        """
        Apply the filter. Create a zeros grid to store the routed rivers.
        Create a sliding window to iterate over ones on mask rivers and
        another sliding window to get values from DEM of reference. Get the
        minimum value from the DEM and the indices of the windows equal to
        this minimum values, all these elements will be new values of rivers.
        Minimum values in DEM are set with a high value to not interfere with
        neighbours.

        Parameters
        ----------
        mask_rivers: ndarray
            mask rivers to iterate over elements with value 1

        Returns
        -------
        ndarray
            Mask with routed rivers
        """
        left_up = self.window_size // 2
        rivers_routed = np.zeros(self.dem.shape)
        sliding = SlidingWindow(mask_rivers, window_size=self.window_size,
                                iter_over_ones=True)
        dem_sliding = SlidingWindow(self.dem, window_size=self.window_size)
        for _, (j, i) in sliding:
            window_dem = dem_sliding[j, i]
            neighbour_min = np.amin(window_dem)
            indices_min = np.nonzero(window_dem == neighbour_min)
            for min_j, min_i in zip(indices_min[0], indices_min[1]):
                indices = (j - left_up + min_j, i - left_up + min_i)
                rivers_routed[indices] = 1
                dem_sliding.grid[indices] = 10000
        return rivers_routed


class QuadraticFilter(Filter):
    """
    Apply a quadratic filter of smoothness

    Attributes
    ----------
    window_size : int
        Window size of circular sliding windows used to compute the filter.

    Methods
    -------
    apply
        apply the filter
    """

    def __init__(self, *, window_size):
        """
        Parameters
        ----------
        window_size : int
            Window size of circular sliding windows used to compute the filter.
        """
        self.window_size = window_size

    def apply(self, dem):
        """
        Apply the quadratic filter of smoothness

        Parameters
        ----------
        dem : ndarray
            image to apply the quadratic smoothness filter

        Returns
        -------
        Smoothed image
        """
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
    """
    Correct Non Available Number (NaN) values in the image, generally with
    extremely lowest values. Use mean of neighbours inside a window as value
    for the element with NaN problems

    Attributes
    ----------
    window_size : int
        Window size of circular sliding windows used to compute the filter.

    Methods
    -------
    apply
        apply the filter
    """

    def __init__(self, *, window_size=3):
        """
        Parameters
        ----------
        window_size : int
            Window size of circular sliding windows used to compute the filter.
        """
        self.window_size = window_size

    def apply(self, dem):
        """
        Apply the filter. Create a mask of negatives values (negatives values
        are considered as NaN values).
        Create a sliding window that iterates over ones values (on negatives
        mask initially created). Create a sliding window avoiding center value
        to get values from the original image. Remove true NaN values (coming
        from sliding window probably), filter values greater than zero and
        assign to the center value (potential NaN value) the mean of
        neighbours non NaN and greater than zero).

        Parameters
        ----------
        dem : ndarray
            Image to correct NaN values.

        Returns
        -------
        The image corrected, the values are modified on the same input image.
        No copy are done with this filter
        """
        mask_nan = MaskNegatives().apply(dem)
        sliding_nans = SlidingWindow(mask_nan, window_size=self.window_size,
                                     iter_over_ones=True)
        dem_sliding = NoCenterWindow(dem, window_size=self.window_size)
        for _, center in sliding_nans:
            neighbours_of_nan = dem_sliding[center]
            neighbours_of_nan = neighbours_of_nan[~np.isnan(neighbours_of_nan)]
            neighbours_positives = neighbours_of_nan[neighbours_of_nan >= 0]
            dem[center] = neighbours_positives.mean()
        return dem


class IsolatedPoints(Filter):
    """
    Remove isolated pixels from a mask. That is, elements in the sliding
    window with no neighbours  are converted to zero.

    Attributes
    ----------
    window_size : int
        Window size of circular sliding windows used to compute the filter.

    Methods
    -------
    apply
        apply the filter
    """

    def __init__(self, *, window_size):
        """
        Parameters
        ----------
        window_size : int
            Window size of circular sliding windows used to compute the filter.
        """
        self.window_size = window_size

    def apply(self, image_to_filter):
        """
        Apply the filter. Create a sliding window with no center element and
        iterating over ones and check if the window contain any element with 1

        Parameters
        ----------
        image_to_filter : ndarray
            Image to remove isolated point

        Returns
        -------
        ndarray
            The image corrected, the values are modified on the same input image.
        No copy are done with this filter
        """
        sliding = NoCenterWindow(image_to_filter, window_size=self.window_size,
                                 iter_over_ones=True)
        for window, center in sliding:
            window = window[~np.isnan(window)]
            image_to_filter[center] = 1. if np.any(window > 0) else 0.
        return image_to_filter


class BlanksFourier(Filter):
    """
    Detect brilliant points in a fourier transform and convert to a binary
    mask. Modified the original image (fourier transform) elements detected
    for the next iteration.

    Attributes
    ----------
    window_size : int
        Window size of circular sliding windows used to compute the filter.

    Methods
    -------
    apply
        apply the filter
    """

    def __init__(self, *, window_size):
        """
        Parameters
        ----------
        window_size : int
            Window size of circular sliding windows used to compute the filter.
        """
        self.window_size = window_size

    def apply(self, image_to_filter):
        """
        Apply the filter. Create a zeros ndarray to store the blanks detected.
        Use a sliding windows that ignores borders and has a inner window to
        prevent very close neighbours interfere with each other. If the center
        element is higher than neighbor mean in 4 times, the element is set as
        a brilliant point. The original element of the image is modified to no
        interfere with other neighbour in another iteration.

        Parameters
        ----------
        image_to_filter : ndarray
            Fourier transform where brilliant points can be present

        Returns
        -------
        tuple(ndarray, ndarray)
            Brilliant points detected (blanks) and the oirginal image
            modified, the elements founded as blank
        """
        filtered_image = np.zeros(image_to_filter.shape)
        fourier_transform = \
            IgnoreBorderInnerSliding(image_to_filter,
                                     window_size=self.window_size,
                                     inner_size=5)
        for window, center in fourier_transform:
            mean_neighbor = np.nanmean(window)
            real_center = tuple(map(lambda x: x - self.window_size // 2,
                                    center))
            if image_to_filter[real_center] > (4 * mean_neighbor):
                filtered_image[real_center] = 1.0
        image_modified = image_to_filter * (1 - filtered_image)
        return filtered_image, image_modified


class MaskNegatives(ComposedFilter):
    """
    Get a mask of Negatives Values. The image will have 1 if the value is
    negative 0 otherwise

    Attributes
    ----------
    filters : list(Filter)
        List of filter to apply in a sequential chain. First creating a
        boolean mask of values lower than zero, then converting this boolean
        values to integer (binary) values.

    Notes
    -----
    Because this class is a subclass of ComposedFilter, the method "apply" of
    the super class is prepared to compute filters in the list filters. Filter
    included in this list must be subclasses of Filter and must have
    implemented the method apply.
    """
    def __init__(self):
        self.filters = [LowerThan(value=0.0), BooleanToInteger()]


class MaskPositives(ComposedFilter):
    """
    Get a mask of Positives Values. The image will have 1 if the value is
    positive 0 otherwise

    Attributes
    ----------
    filters : list(Filter)
        List of filter to apply in a sequential chain. First creating a
        boolean mask of values greather than zero, then converting this boolean
        values to integer (binary) values.

    Notes
    -----
    Because this class is a subclass of ComposedFilter, the method "apply" of
    the super class is prepared to compute filters in the list filters. Filter
    included in this list must be subclasses of Filter and must have
    implemented the method apply.
    """

    def __init__(self):
        self.filters = [GreaterThan(value=0.0), BooleanToInteger()]


class MaskTallGroves(ComposedFilter):
    """
    Get a mask of values greather than 1.5. The image will have 1 if the value
    is greather than 1.5, 0 otherwise.

    Attributes
    ----------
    filters : list(Filter)
        List of filter to apply in a sequential chain. First creating a
        boolean mask of values greather than 1.5, then converting this boolean
        values to integer (binary) values.

    Notes
    -----
    Because this class is a subclass of ComposedFilter, the method "apply" of
    the super class is prepared to compute filters in the list filters. Filter
    included in this list must be subclasses of Filter and must have
    implemented the method "apply".
    """

    def __init__(self):
        self.filters = [GreaterThan(value=1.5), BooleanToInteger()]


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
    Apply the chain of filters sequentially to:
    Detect blanks fourier
    Remove isolated points
    Expands blanks detected.

    Attributes
    ----------
    filters : list(Filter)
        List of filter to apply in a sequential chain. This composed filter is
        set to: First detecting blanks in fourier transform, then removing
        isolated points and finishing expanding the blanks detected

    Notes
    -----
    Because this class is a subclass of ComposedFilter, the method "apply" of
    the super class is prepared to compute filters in the list filters. Filters
    included in this list of filters must be subclass of Filter and must have
    implemented the method apply.
    """
    def __init__(self):
        self.filters = [DetectBlanksFourier(), IsolatedPoints(window_size=3),
                        ExpandFilter(window_size=13)]


class TidyingLagoons(ComposedFilter):
    """
    Apply the chain of filters sequentially to tidy the shape of lagoons:
    Erode borders of detected lagoons
    Expand eroded lagoons
    Multiply with the origin image to filter
    Dilate the result to cover the borders of lagoons

    Attributes
    ----------
    filters : list(Filter)
        List of filter to apply in a sequential chain. This composed filter is
        set to: Erode borders, expand eroded lagoons, mupltiply with the
        origin mask, dilate to cover borders

    Notes
    -----
    Because this class is a subclass of ComposedFilter, the method "apply" of
    the super class is prepared to compute filters in the list filters. Filters
    included in this list of filters must be subclass of Filter and must have
    implemented the method apply.
    """

    def __init__(self):
        self.filters = [BinaryErosion(iterations=2),
                        ExpandFilter(window_size=7),
                        ProductFilter(),
                        GreyDilation(size=(7, 7))]

    def apply(self, image_to_filter):
        """
        Apply the filter. First of all, save the original mask to the product
         filter attribute

        Parameters
        ----------
        image_to_filter : ndarray
            Initial lagoons detected

        Returns
        -------
            Lagoons tidied
        """
        self.filters[2].factor = content = image_to_filter
        for filter_ in self.filters:
            content = filter_.apply(content)
        return content


class LagoonsDetection(ComposedFilterResults):
    def __init__(self):
        super().__init__()
        self.filters = [CorrectNANValues(), MajorityFilter(window_size=11),
                        TidyingLagoons(), MaskPositives()]

    def apply(self, image_to_filter):
        result = super().apply(image_to_filter)
        self.hsheds_nan_fixed = self.results["CorrectNANValues"]
        self.mask_lagoons = self.results["MaskPositives"]
        self.lagoons_values = self.results["TidyingLagoons"]
        return result


class GrovesCorrection(ComposedFilter):

    def __init__(self, groves_class):
        self.partial_results = []
        self.filters = [QuadraticFilter(window_size=15), SubtractionFilter(),
                        MaskTallGroves(), ProductFilter(factor=groves_class),
                        SubtractionFilter(minuend=1)]

    def apply(self, image_to_filter):
        self.filters[1].minuend = content = image_to_filter
        for filter_ in self.filters:
            content = filter_.apply(content)
            self.partial_results.append(copy.deepcopy(content))
        final_addition = AdditionFilter(adding=self.partial_results[0])
        final_product = ProductFilter(factor=self.partial_results[1])
        self.filters = [final_product, final_addition]
        return super().apply(content)


class GrovesCorrectionsIter(ComposedFilter):

    # TODO: Ver de modificar esto
    def __init__(self, groves_class):
        self.filters = [GrovesCorrection(groves_class),
                        GrovesCorrection(groves_class),
                        GrovesCorrection(groves_class)]


class ProcessRivers(ComposedFilter):

    def __init__(self, hsheds):
        self.filters = [MaskPositives(), ExpandFilter(window_size=3),
                        RouteRivers(window_size=3, dem=hsheds),
                        BinaryClosing()]


class ClipLagoonsRivers(ComposedFilter):

    def __init__(self, mask_lagoons, rivers_routed_closing):
        self.filters = [ProductFilter(factor=mask_lagoons), MaskPositives(),
                        BitwiseXOR(operand=rivers_routed_closing)]


class FourierInitial(ComposedFilterResults):
    def __init__(self):
        super().__init__()
        self.filters = [FourierTransform(), FourierShift(), AbsolutValues()]

    def apply(self, image_to_filter):
        result = super().apply(image_to_filter)
        self.fourier_shift = self.results["FourierShift"]
        return result



class FourierProcessQuarters(Filter):

    def __init__(self, fft_transform_abs):
        self._filters = [self._get_firsts_quarters,
                         self._apply_mask_fourier,
                         self._fill_complete_quarters,
                         self._getting_reversed_masks,
                         self._fill_complete_mask]
        self.fft_transform_abs = fft_transform_abs
        self._ny, self._nx = fft_transform_abs.shape
        self._mid_y, self._y_odd = divmod(self._ny, 2)
        self._mid_x, self._x_odd = divmod(self._nx, 2)
        self._margin = 10

    def apply(self, *args):
        content = None
        for filter_ in self._filters:
            content = filter_(content)
        return content

    def _get_firsts_quarters(self, args=None):
        """
        Get first (upper left) and second (upper right) quarter from Fourier
        transform image, removing central strip with a .
        """
        fst_quarter_fourier = \
            self.fft_transform_abs[:self._mid_y - self._margin,
            :self._mid_x - self._margin]
        snd_quarter_fourier = \
            self.fft_transform_abs[:self._mid_y - self._margin,
            self._mid_x + self._margin +
            self._x_odd:self._nx]
        return fst_quarter_fourier, snd_quarter_fourier

    def _apply_mask_fourier(self, quarters):
        """
        Apply detection blanks fourier algorithm to first two quarters of the
        image.

        Parameters
        ----------
        quarters: tuple(ndarray, ndarray)
            images to apply fourier blank detection

        Returns
        -------
            tuple(ndarray, ndarray)
                tuple of masks of blank fourier detected for each quarter
        """
        mask_quarters = tuple(map(MaskFourier().apply, quarters))
        return mask_quarters

    def _fill_complete_quarters(self, quarters):
        """
        Create quarters of base image with zeros values (these quarters
        contains also margin and central line). After creation these quarters
        are filled with quarters input. They suposed to be mask of blank fourier
        detected previously.

        Parameters
        ----------
        quarters: tuple(ndarray, ndarray)
            quarters that are going to be place inside the quarter of real size
            the image

        Returns
        -------
        first and second quarters of real size of the image filled with mask
        fourier detected.
        """
        fst_complete_quarter = np.zeros((self._mid_y, self._mid_x))
        snd_complete_quarter = np.zeros((self._mid_y, self._mid_x))
        fst_complete_quarter[:self._mid_y - self._margin,
        :self._mid_x - self._margin] = quarters[0]
        snd_complete_quarter[:self._mid_y - self._margin,
        self._margin:self._mid_x] = quarters[1]
        return fst_complete_quarter, snd_complete_quarter

    def _getting_reverse_indices(self):
        """
        Get indices mesh of images reversed

        Returns
        -------
            Combination of pairs of indices to reverse some image.
        """
        reverse_x = (self._mid_x - 1) - np.arange(self._mid_x)
        reverse_y = (self._mid_y - 1) - np.arange(self._mid_y)
        indices = np.ix_(reverse_y, reverse_x)
        return indices

    def _getting_reversed_masks(self, quarters):
        """
        Create images reversed of input images.

        Parameters
        ----------
        quarters: tuple(ndarray, ndarray)
            images to be reversed

        Returns
        -------
            tuple(ndarray, ndarray, ndarray, ndarray)
                tuple of four images, the first two are the input images, while
                the second ones are the reversed images.
        """
        indices = self._getting_reverse_indices()
        trd_complete_quarter = quarters[1][indices]
        fth_complete_quarter = quarters[0][indices]

        return (quarters[0], quarters[1], trd_complete_quarter,
                fth_complete_quarter)

    def _fill_complete_mask(self, quarters):
        """
        Assemble the complete fourier mask using quarters provided as input

        Parameters
        ----------
        quarters: tuple(ndarray, ndarray, ndarray, ndarray)
            images needed to assemble the final fourier mask.
        Returns
        -------
            ndarray
                fourier mask complete.
        """
        masks_fourier = np.zeros((self._ny, self._nx))
        masks_fourier[:self._mid_y, :self._mid_x] = quarters[0]
        masks_fourier[:self._mid_y,
        self._mid_x + self._x_odd:self._nx] = quarters[1]
        masks_fourier[self._mid_y + self._y_odd:self._ny,
        :self._mid_x] = quarters[2]
        masks_fourier[self._mid_y + self._y_odd:self._ny,
        self._mid_x + self._x_odd:self._nx] = quarters[3]
        return masks_fourier


class DetectApplyFourier(ComposedFilter):
    def __init__(self):
        self.initial = FourierInitial()

    def apply(self, image_to_filter):
        self.fft_transform_abs = self.initial.apply(image_to_filter)
        self.filters = [FourierProcessQuarters(self.fft_transform_abs),
                        SubtractionFilter(minuend=1),
                        ProductFilter(factor=self.initial.fourier_shift),
                        FourierIShift(), FourierITransform(), AbsolutValues()]
        return super().apply(image_to_filter)


class PostProcessingFinal(ComposedFilter):

    def __init__(self):
        self.filters = [Convolve(), Around()]
