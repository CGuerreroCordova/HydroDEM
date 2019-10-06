"""
Provide the abstract class for filters. Base classes to define filter classes.
"""

from abc import ABC, abstractmethod
from numpy import ndarray
from exceptions import NumpyArrayExpectedError


class Filter(ABC):
    """
    Represent a basic filter with abstract method apply to implement for
    subclasses. The calling to this super class method for subclasses check
    correctness type of input to apply

    Methods
    -------
    apply
        Apply the filter to the input image. The filter definition must be
        placed in this function
    """

    @abstractmethod
    def apply(self, image_to_filter):
        """
        Apply the filter

        Parameters
        ----------
        image_to_filter: ndarray
            image grid to apply the filter

        Raises
        ------
        NumpyArrayExpectedError
            If image_to_filter provided is not ndarray type
        """
        # TODO: Check if it necesary include NotImplementedError
        if not isinstance(image_to_filter, ndarray):
            raise NumpyArrayExpectedError(image_to_filter)


class ComposedFilter(Filter):
    """
    Represent a composed filter. A Composed filter is a chain sequence of
    filters. The definition of the filters chain is defined on attribute
    filters.

    Attributes
    ----------
    filters : list(Filter)
        Chain sequence of filters to apply sequentially
    """

    def __init__(self):
        """
        Composed chain of filters definition must be placed in the attribute
        filters.
        """
        self.filters = []

    def apply(self, image_to_filter):
        """
        Apply sequentially chain of filters. Initial image must be provided as
        parameter. Correctness type is checked calling the super class method.

        Parameters
        ----------
        image_to_filter: ndarray
            initial image to begin the chain sequential of filter application.

        Returns
        -------
        Final result of image filtered after passing through the chain of
        filters
        """
        super().apply(image_to_filter)
        content = image_to_filter
        for filter_ in self.filters:
            content = filter_.apply(content)
        return content


class ComposedFilterResults(Filter):
    """
    The same as ComposedFilter class but saving intermediate results in an
    attribute dict for  specific purpose of the subclass filter definition

    Attributes
    ----------
    filters : list(Filter)
        Chain sequence of filters to apply sequentially
    results : dict
        Store intermediate results belonging to chain filters application.
    """

    def __init__(self):
        """
        Composed chain of filters definition must be placed in the attribute
        filters. Intermediate results of filter application will be saved in
        results dictionary.
        """
        self.filters = []
        self.results = dict()

    def apply(self, image_to_filter):
        """
        Apply sequentially chain of filters. Each result of filter application
        belonging to chain of filters is saved in attribute results. Initial
        image must be provided as parameter. Correctness type is checked
        calling the super class method.

        Parameters
        ----------
        image_to_filter: ndarray
            initial image to begin the chain sequential of filter application.

        Returns
        -------
        Final result of image filtered after passing through the chain of
        filters
        """
        super().apply(image_to_filter)
        content = image_to_filter
        for filter_ in self.filters:
            content = filter_.apply(content)
            self.results[filter_.__class__.__name__] = content
        return content
