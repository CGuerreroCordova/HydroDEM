"""
Provide simple operation filters: Logical, Mathematical
"""
from filters import Filter


class LowerThan(Filter):
    """
    Filter the values from an image that are lower than a specific value.
    Return an image with binary values (o or 1, integer)

    Attributes
    ----------
    value : int, float
        The value to compare the elements of the image.

    Methods
    -------
    apply
        Apply the filter. Check input data has the valid type to apply the
        filter.
    """

    def __init__(self, *, value):
        """
        Set the value for which the values of the image will be compared

        Parameters
        ----------
        value : int, float
            The value to compare the elements of the image.
        """
        self.value = value

    def apply(self, image_to_filter):
        """
        Apply the filter. Check the type of the image input parameter.

        Parameters
        ----------
        image_to_filter: ndarray
            Image to apply the filter

        Returns
        -------
        Filtered image, where elements are lower than value will be 1 or True,
        0 or False otherwise
        """
        super().apply(image_to_filter)
        return image_to_filter < self.value


class GreaterThan(Filter):
    """
    Filter the values from an image that are greater than a specific value.
    Return an image with binary values (o or 1, integer)

    Attributes
    ----------
    value : int, float
        The value to compare the elements of the image.

    Methods
    -------
    apply
        Apply the filter. Check input data has the valid type to apply the
        filter.
    """

    def __init__(self, *, value):
        """
        Set the value for which the values of the image will be compared

        Parameters
        ----------
        value : int, float
            The value to compare the elements of the image.
        """
        self.value = value

    def apply(self, image_to_filter):
        """
        Apply the filter. Check the type of the image input parameter.

        Parameters
        ----------
        image_to_filter: ndarray
            Image to apply the filter

        Returns
        -------
        Filtered image, where elements are greater than value will be 1 or
        True, 0 or False otherwise
        """
        super().apply(image_to_filter)
        return image_to_filter > self.value


class BooleanToInteger(Filter):
    """
    Convert the values of an image grid from boolean to integer. This is
    reached multiplying the boolean values for integer 1. True values will be
    converted to 1, False values to 0. This is necesarry to get a mask of
    values

    Methods
    -------
    apply
        Apply the filter. Multiply each element of the image by one. Check
        input data has the valid type to apply the filter.
    """

    def apply(self, image_to_filter):
        """
        Apply the filter. Check the type of the image input parameter. Because
        the expected input value is a ndarray image, multiplying this object by
        1, it implies to multiply each element of the array by 1, so converting
        the value from boolean type to integer type.

        Parameters
        ----------
        image_to_filter: ndarray
            Image to apply the filter

        Returns
        -------
        Filtered image, where elements now have integer valueas instead of
        boolean values
        """
        super().apply(image_to_filter)
        return image_to_filter * 1


class ProductFilter(Filter):
    """
    Allow the multiplication between grids, one to one elements.

    Attributes
    ----------
    factor : int, float
        One of the factors to be multiplied. If an scalar value is passed
        as parameter, this value is multiplied by each element of the other
        factor (default value 1)

    Methods
    -------
    apply
        Apply the filter. Check input data has the valid type to apply the
        filter.
    """

    def __init__(self, factor=1):
        """
        Set one of the factors to be multiplied,.

        Parameters
        ----------
        factor : int, float
            One of the factors to be multiplied. If an scalar value is passed
            as parameter, this value is multiplied by each element of the other
            factor (default value 1)
        """
        self.factor = factor

    def apply(self, image_to_filter):
        """
        Apply the filter. Check the type of the image input parameter. Multiply
        the value in the attributes by the argument passed as parameter.

        Parameters
        ----------
        image_to_filter: ndarray
            Image to apply the filter

        Returns
        -------
        Product as result of multiplication of factors.
        """
        super().apply(image_to_filter)
        return self.factor * image_to_filter


class AdditionFilter(Filter):
    """
    Allow the addition between grids, one to one elements.

    Attributes
    ----------
    addend : int, float
        One of the addends to be addend. If an scalar value is passed
        as parameter, this value is added to each element of the other
        addend (default value 0)

    Methods
    -------
    apply
        Apply the filter. Check input data has the valid type to apply the
        filter.
    """

    def __init__(self, addend=0):
        """
        Set one of the addends to be added.

        Parameters
        ----------
        addend : int, float
            One of the addends to be addend. If an scalar value is passed
            as parameter, this value is added to each element of the other
            addend (default value 0)
        """
        self.addend = addend

    def apply(self, image_to_filter):
        """
        Apply the filter. Check the type of the image input parameter. Add
        the value in the attributes with the argument passed as parameter.

        Parameters
        ----------
        image_to_filter: ndarray
            Image to apply the addition

        Returns
        -------
        Addition as result of addends.
        """
        super().apply(image_to_filter)
        return self.addend + image_to_filter


class SubtractionFilter(Filter):
    """
    Allow the subtraction between grids, one to one elements.

    Attributes
    ----------
    minuend : int, float
        Minuend. If an scalar value is passed as parameter, this value is
        on which the substraction will be applied (default value 0.0)

    Methods
    -------
    apply
        Apply the filter. Check input data has the valid type to apply the
        filter.
    """

    def __init__(self, *, minuend=0.0):
        """
        Set the minuend.

        Parameters
        ----------
        minuend : int, float
            Minuend. If an scalar value is passed as parameter, this value is
            on which the substraction will be applied (default value 0.0)
        """
        self.minuend = minuend

    def apply(self, subtracting):
        """
        Apply the filter. Check the type of the image input parameter.
        Substract subtracting from minuend.

        Parameters
        ----------
        subtracting: ndarray
            Image to subtract from the minuend

        Returns
        -------
        Grid with the result of subtraction.
        """
        return self.minuend - subtracting

