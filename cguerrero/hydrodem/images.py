from abc import ABC, abstractmethod

import gdal
import numpy as np

from filters import LagoonsDetection, ClipLagoonsRivers, \
    ProcessRivers, DetectApplyFourier, BinaryClosing, GrovesCorrectionsIter
from utils_dem import resample_and_cut, unzip_resource, rasterize_rivers, \
    clip_lines_vector
from settings import (TREE_CLASS_INPUT_ZIP, SRTM_FILE_INPUT_ZIP,
                      TREE_CLASS_INPUT, AREA_OVER, TREE_CLASS_AREA,
                      SRTM_AREA_OVER, SRTM_FILE_INPUT, HSHEDS_FILE_INPUT_ZIP,
                      HSHEDS_FILE_TIFF, HSHEDS_FILE_INPUT, HSHEDS_AREA_OVER,
                      RIVERS_SHAPE, RIVERS_TIF, RIVERS_FULL, RIVERS_ZIP)


class Image(ABC):

    @abstractmethod
    def prepare(self):
        raise NotImplementedError

    @abstractmethod
    def process(self):
        raise NotImplementedError


class SRTM(Image):
    def __init__(self):
        self.fourier = Fourier()
        self.groves = Groves()

    def prepare(self):
        pass

    def process(self):
        self.fourier.prepare()
        fourier_corrected = self.fourier.process()
        self.groves.prepare()
        return self.groves.process(fourier_corrected)


class Fourier(Image):

    def prepare(self):
        unzip_resource(SRTM_FILE_INPUT_ZIP)
        resample_and_cut(SRTM_FILE_INPUT, AREA_OVER, SRTM_AREA_OVER)

    def process(self):
        srtm_raw = gdal.Open(SRTM_AREA_OVER).ReadAsArray()
        return DetectApplyFourier().apply(srtm_raw)


class Groves(Image):

    def prepare(self):
        unzip_resource(TREE_CLASS_INPUT_ZIP)
        resample_and_cut(TREE_CLASS_INPUT, AREA_OVER, TREE_CLASS_AREA)
        tree_class_raw = gdal.Open(TREE_CLASS_AREA).ReadAsArray()
        return BinaryClosing(structure=np.ones((3, 3))).apply(tree_class_raw)

    def process(self, srtm_to_process):
        tree_class = self.prepare()
        return GrovesCorrectionsIter(tree_class).apply(srtm_to_process)


class HSHEDS(Image):

    def __init__(self):
        self.lagoons = Lagoons()
        self.rivers = Rivers()

    def prepare(self):
        unzip_resource(HSHEDS_FILE_INPUT_ZIP)
        gdt_options = gdal.TranslateOptions(format='GTIFF')
        gdal.Translate(HSHEDS_FILE_TIFF, HSHEDS_FILE_INPUT,
                       options=gdt_options)
        resample_and_cut(HSHEDS_FILE_TIFF, AREA_OVER, HSHEDS_AREA_OVER)
        return self

    def process(self):
        lagoons = self.lagoons.process()
        rivers = self.rivers.process(lagoons)
        return lagoons, rivers


class Lagoons(Image):

    def prepare(self):
        pass

    def process(self):
        hydro_sheds = gdal.Open(HSHEDS_AREA_OVER).ReadAsArray()
        lagoons = LagoonsDetection()
        lagoons.apply(hydro_sheds)
        return lagoons


class Rivers(Image):

    def prepare(self):
        unzip_resource(RIVERS_ZIP)
        clip_lines_vector(RIVERS_FULL, AREA_OVER, RIVERS_SHAPE)
        rasterize_rivers(RIVERS_SHAPE, RIVERS_TIF)

    def process(self, lagoons):
        self.prepare()
        rivers = gdal.Open(RIVERS_TIF).ReadAsArray()
        rivers_routed = ProcessRivers(lagoons.hsheds_nan_fixed).apply(rivers)
        return ClipLagoonsRivers(lagoons.mask_lagoons,
                                 rivers_routed).apply(rivers_routed)
