
import gdal
import numpy as np

from image import Image
from filters.custom_filters import (DetectApplyFourier, BinaryClosing,
                                    GrovesCorrectionsIter)
from utils_dem import (resample_and_cut, unzip_resource)
from config_loader import Config


class SRTM(Image):

    def __init__(self, area_of_interest):
        super().__init__(area_of_interest)
        self.srtm_zip = Config.srtm('SRTM_ZIP')
        self.srtm_tif = Config.srtm('SRTM_TIF')
        self.srtm_interest = Config.srtm('SRTM_AREA')
        self.fourier = Fourier(self.srtm_interest)
        self.groves = Groves(area_of_interest)

    def _prepare(self):
        unzip_resource(self.srtm_zip)
        resample_and_cut(self.srtm_tif, self.aoi, self.srtm_interest)

    def process(self):
        self._prepare()
        fourier_corrected = self.fourier.process()
        return self.groves.process(fourier_corrected)


class Fourier(Image):

    def __init__(self, area_of_interest):
        self.srtm_interest = area_of_interest

    def process(self):
        srtm_raw = gdal.Open(self.srtm_interest).ReadAsArray()
        return DetectApplyFourier().apply(srtm_raw)


class Groves(Image):

    def __init__(self, area_of_interest):
        super().__init__(area_of_interest)
        self.groves_class_zip = Config.groves('GROVES_ZIP')
        self.groves_class_tif = Config.groves('GROVES_TIF')
        self.groves_interest = Config.groves('GROVES_AREA')

    def _prepare(self):
        unzip_resource(self.groves_class_zip)
        resample_and_cut(self.groves_class_tif, self.aoi, self.groves_interest)
        groves_class_raw = gdal.Open(self.groves_interest).ReadAsArray()
        return BinaryClosing(structure=np.ones((3, 3))).apply(groves_class_raw)

    def process(self, srtm):
        groves_class = self._prepare()
        return GrovesCorrectionsIter(groves_class, iterations=3).apply(srtm)
