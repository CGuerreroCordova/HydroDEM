import copy
from numpy import bitwise_xor, abs, around, ones
from scipy.ndimage import (binary_erosion, binary_closing, grey_dilation,
                           convolve)
from scipy import fftpack
from filters import Filter


class BitwiseXOR(Filter):

    def __init__(self, *, operand):
        self.operand = copy.deepcopy(operand)

    def apply(self, image_to_filter):
        return bitwise_xor(self.operand, image_to_filter)


class BinaryErosion(Filter):
    """
    Represents the Mmajority Filter
    """

    def __init__(self, *, iterations):
        self.iterations = iterations

    def apply(self, image_to_filter):
        return binary_erosion(image_to_filter, iterations=self.iterations)


class BinaryClosing(Filter):
    """
    Represents the Mmajority Filter
    """

    def __init__(self, *, structure=None):
        self.structure = structure

    def apply(self, image_to_filter):
        return binary_closing(image_to_filter, structure=self.structure)


class GreyDilation(Filter):

    def __init__(self, *, size):
        self.size = size

    def apply(self, image_to_filter):
        return grey_dilation(image_to_filter, size=self.size)


class FourierTransform(Filter):

    def apply(self, image_to_filter):
        return fftpack.fft2(image_to_filter)


class FourierITransform(Filter):

    def apply(self, image_to_filter):
        return fftpack.ifft2(image_to_filter)


class FourierShift(Filter):

    def apply(self, image_to_filter):
        return fftpack.fftshift(image_to_filter)


class FourierIShift(Filter):

    def apply(self, image_to_filter):
        return fftpack.ifftshift(image_to_filter)


class AbsolutValues(Filter):

    def apply(self, image_to_filter):
        return abs(image_to_filter)


class Around(Filter):

    def apply(self, image_to_filter):
        return around(image_to_filter)


class Convolve(Filter):

    def __init__(self, weights=ones((3, 3))):
        self.weights = weights

    def apply(self, image_to_filter):
        return convolve(image_to_filter, weights=self.weights) / \
               self.weights.size
