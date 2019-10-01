from abc import ABC, abstractmethod


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


class ComposedFilterResults(Filter):

    def __init__(self):
        self.results = dict()

    def apply(self, image_to_filter):
        content = image_to_filter
        for filter in self.filters:
            content = filter.apply(content)
            self.results[filter.__class__.__name__] = content
        return content
