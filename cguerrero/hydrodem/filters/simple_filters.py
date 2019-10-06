from filters import Filter


class LowerThan(Filter):

    def __init__(self, *, value):
        self.value = value

    def apply(self, image_to_filter):
        super().apply(image_to_filter)
        return image_to_filter < self.value


class GreaterThan(Filter):

    def __init__(self, *, value):
        self.value = value

    def apply(self, image_to_filter):
        super().apply(image_to_filter)
        return image_to_filter > self.value


class BooleanToInteger(Filter):

    def apply(self, image_to_filter):
        super().apply(image_to_filter)
        return image_to_filter * 1


class ProductFilter(Filter):

    def __init__(self, factor=1):
        self.factor = factor

    def apply(self, image_to_filter):
        super().apply(image_to_filter)
        return self.factor * image_to_filter


class AdditionFilter(Filter):

    def __init__(self, adding=0):
        self.adding = adding

    def apply(self, adding):
        super().apply(adding)
        return self.adding + adding


class SubtractionFilter(Filter):

    def __init__(self, *, minuend=0.0):
        self.minuend = minuend

    def apply(self, subtracting):
        return self.minuend - subtracting

