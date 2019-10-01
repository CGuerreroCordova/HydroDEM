from abc import ABC, abstractmethod


class Image(ABC):
    def __init__(self, area_of_interest):
        self.aoi = area_of_interest

    @abstractmethod
    def process(self):
        raise NotImplementedError
