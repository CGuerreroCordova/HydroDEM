from filters import Filter


class LowerThan(Filter):

    def __init__(self, *, value):
        self.value = value

    def apply(self, image_to_filter):
        # TODO: Condition about values, Exceptions
        return image_to_filter < self.value


class GreaterThan(Filter):

    def __init__(self, *, value):
        self.value = value

    def apply(self, image_to_filter):
        # TODO: Condition about values, Exceptions
        return image_to_filter > self.value


class BooleanToInteger(Filter):

    def apply(self, image_to_filter):
        # TODO: conditions about type of values
        return image_to_filter * 1


class ProductFilter(Filter):

    def __init__(self, factor=1):
        # TODO Conditions about types.
        self.factor = factor

    def apply(self, factor):
        return self.factor * factor


class AdditionFilter(Filter):

    def __init__(self, adding=0):
        # TODO Conditions about types.
        self.adding = adding

    def apply(self, adding):
        return self.adding + adding


class SubtractionFilter(Filter):

    def __init__(self, *, minuend=0.0):
        self.minuend = minuend

    def apply(self, subtracting):
        return self.minuend - subtracting
