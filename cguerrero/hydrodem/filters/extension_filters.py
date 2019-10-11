"""
Provide a wrap for already implemented filters in external libraries.
"""
import copy
from numpy import bitwise_xor, abs, around, ones
from scipy.ndimage import (binary_erosion, binary_closing, grey_dilation,
                           convolve)
from scipy import fftpack
from filters import Filter


class BitwiseXOR(Filter):
    """
    Provide a wrap for Bitwise-Xor numpy filter

    Methods
    -------
    apply
        Apply the wrapped filter

    See Also
    --------
    numpy.bitwise_xor: for detailed information
    """

    def __init__(self, *, operand):
        """
        Initialize the filter with one of the operands of the filter. Make a
        copy of the filter

        Parameters
        ----------
        operand : ndarray
            First operand to apply the filter
        """
        self.operand = copy.deepcopy(operand)

    def apply(self, image_to_filter):
        """
        Apply the filter calling the already implemented filter in numpy
        library. For more details about this filter you can see documentation
        in numpy library

        Parameters
        ----------
        image_to_filter : ndarray
            Image to apply the filter, this is the second operand of the filter

        Returns
        -------
        ndarray
            Result of applying filter using operands
        """
        return bitwise_xor(self.operand, image_to_filter)


class AbsoluteValues(Filter):
    """
    Provide a wrap to apply the absolute function of numpy library.
    Calculate the absolute value element-wise.

    Methods
    -------
    apply
        Apply the wrapped filter

    See Also
    --------
    numpy.absolute: for detailed information
    """

    def apply(self, image_to_filter):
        """
        Apply the filter calling the already implemented absolute function in
        numpy library. For more details about this filter you can see
        documentation in numpy library.

        Parameters
        ----------
        image_to_filter : ndarray
            Image to apply the filter.

        Returns
        -------
        ndarray
            Result of applying filter
        """
        return abs(image_to_filter)


class Around(Filter):
    """
    Provide a wrap to apply the absolute function of numpy library.
    Calculate the absolute value element-wise.

    Methods
    -------
    apply
        Apply the wrapped filter

    See Also
    --------
    numpy.absolute: for detailed information
    """

    def apply(self, image_to_filter):
        """
        Apply the filter calling the already implemented around function in
        numpy library. For more details about this filter you can see
        documentation in numpy library.

        Parameters
        ----------
        image_to_filter : ndarray
            Image to apply the filter.

        Returns
        -------
        ndarray
            Result of applying filter
        """
        return around(image_to_filter)


class Convolve(Filter):
    """
    Provide a wrap to apply the convolve function of scipy library.
    Multidimensional convolution. The array is convolved with the given kernel.


    Methods
    -------
    apply
        Apply the wrapped convolve filter

    See Also
    --------
    scipy.convolve: for detailed information
    """

    def __init__(self, weights=ones((3, 3))):
        """
        Initialize the filter with the weights to apply the convolve filter

        Parameters
        ----------
        weights : array_like, optional
            Array of weights, same number of dimensions as input
            (default is ones((3, 3)))
        """
        self.weights = weights

    def apply(self, image_to_filter):
        """
        Apply the filter calling the already implemented convolve filter in
        scipy library. For more details about this filter you can see
        documentation in scipy library.

        Parameters
        ----------
        image_to_filter : ndarray
            Image to apply the filter.

        Returns
        -------
        ndarray
            Result of applying filter
        """
        return convolve(image_to_filter,
                        weights=self.weights) / self.weights.size


class BinaryErosion(Filter):
    """
    Provide a wrap for Binary Erosion scipy filter

    Methods
    -------
    apply
        Apply the wrapped filter

    See Also
    --------
    scipy.ndimage.binary_erosion: for detailed information
    """

    def __init__(self, *, iterations):
        """
        Initialize the filter with amount of iterations to be applied the
        filter

        Parameters
        ----------
        iterations : int
            Amount of iterations to apply the filter
        """
        self.iterations = iterations

    def apply(self, image_to_filter):
        """
        Apply the filter calling the already implemented filter in scipy
        library. For more details about this filter you can see documentation
        in numpy library.

        Parameters
        ----------
        image_to_filter : ndarray
            Image to apply the filter.

        Returns
        -------
        ndarray
            Result of applying filter
        """
        return binary_erosion(image_to_filter, iterations=self.iterations)


class BinaryClosing(Filter):
    """
    Provide a wrap for Binary Closing scipy filter

    Methods
    -------
    apply
        Apply the wrapped filter

    See Also
    --------
    scipy.ndimage.binary_closing: for detailed information
    """

    def __init__(self, *, structure=None):
        """
        Initialize the filter with structure to use to apply the filter

        Parameters
        ----------
        structure : array_like, optional
            Structuring element used for the closing. Non-zero elements are
            considered True. If no structuring element is provided an element
            is generated with a square connectivity equal to one (i.e., only
            nearest neighbour are connected to the center, diagonally-connected
            elements are not considered neighbour).
        """
        self.structure = structure

    def apply(self, image_to_filter):
        """
        Apply the filter calling the already implemented filter in scipy
        library. For more details about this filter you can see documentation
        in numpy library.

        Parameters
        ----------
        image_to_filter : ndarray
            Image to apply the filter.

        Returns
        -------
        ndarray
            Result of applying filter
        """
        return binary_closing(image_to_filter, structure=self.structure)


class GreyDilation(Filter):
    """
    Provide a wrap for Grey Dilation scipy filter

    Methods
    -------
    apply
        Apply the wrapped filter

    See Also
    --------
    scipy.ndimage.grey_dilation: for detailed information
    """

    def __init__(self, *, size):
        """
        Initialize the filter with size to use to apply the filter

        Parameters
        ----------
        size : tuple of ints
            Shape of a flat and full structuring element used for the grayscale
            dilation. Optional if `footprint` or `structure` is provided.
        """
        self.size = size

    def apply(self, image_to_filter):
        """
        Apply the filter calling the already implemented filter in scipy
        library. For more details about this filter you can see documentation
        in numpy library.

        Parameters
        ----------
        image_to_filter : ndarray
            Image to apply the filter.

        Returns
        -------
        ndarray
            Result of applying filter
        """
        return grey_dilation(image_to_filter, size=self.size)


class FourierTransform(Filter):
    """
    Provide a wrap to apply 2-D discrete Fourier transform. Return the
    two-dimensional discrete Fourier transform of the 2-D argument `x`.

    Methods
    -------
    apply
        Apply the wrapped filter

    See Also
    --------
    scipy.fftpack.fft2: for detailed information
    """

    def apply(self, image_to_filter):
        """
        Image to apply the Fourier transform

        Parameters
        ----------
        image_to_filter : ndarray
            Image to apply the filter.

        Returns
        -------
        ndarray
            The two-dimensional discrete Fourier transform of the 2-D argument
            `x`.
        """
        return fftpack.fft2(image_to_filter)


class FourierITransform(Filter):
    """
    Provide a wrap to apply 2-D discrete inverse Fourier transform of real or
    complex sequence. Return inverse two-dimensional discrete Fourier transform
    of arbitrary type sequence x.

    Methods
    -------
    apply
        Apply the wrapped filter

    See Also
    --------
    scipy.fftpack.ifft2: for detailed information
    """

    def apply(self, image_to_filter):
        """
        Image to apply the inverse of Fourier transform

        Parameters
        ----------
        image_to_filter : ndarray
            Image to apply the filter.

        Returns
        -------
        ndarray
            Inverse two-dimensional discrete Fourier transform of arbitrary
            type sequence x.
        """
        return fftpack.ifft2(image_to_filter)


class FourierShift(Filter):
    """
    Provide a wrap to shift the zero-frequency component to the center of the
    spectrum.

    Methods
    -------
    apply
        Apply the wrapped filter

    See Also
    --------
    scipy.fftpack.fftshift: for detailed information
    """

    def apply(self, image_to_filter):
        """
        Image to apply the shift filter.

        Parameters
        ----------
        image_to_filter : ndarray
            Image to apply the shift.

        Returns
        -------
        ndarray
            The shifted array
        """
        return fftpack.fftshift(image_to_filter)


class FourierIShift(Filter):
    """
    Provide a wrap to apply the inverse function of shifting the
    zero-frequency component to the center of the spectrum.

    Methods
    -------
    apply
        Apply the wrapped filter

    See Also
    --------
    scipy.fftpack.ifftshift: for detailed information
    """

    def apply(self, image_to_filter):
        """
        Image to apply the inverse function of shift filter.

        Parameters
        ----------
        image_to_filter : ndarray
            Image to apply the inverse of shift function.

        Returns
        -------
        ndarray
            The shifted array
        """
        return fftpack.ifftshift(image_to_filter)
