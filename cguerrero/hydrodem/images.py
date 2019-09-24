from abc import ABC, abstractmethod

import gdal
import numpy as np

from filters import (LagoonsDetection, ClipLagoonsRivers, ProcessRivers,
                     DetectApplyFourier, BinaryClosing, GrovesCorrectionsIter)
from utils_dem import (resample_and_cut, unzip_resource, rasterize_rivers,
                       clip_lines_vector)
from config_loader import Config

# from settings import (GROVES_ZIP, SRTM_ZIP,
#                       GROVES_TIF, AREA_ENVELOPE, GROVES_AREA,
#                       SRTM_AREA, SRTM_TIF, HSHEDS_ZIP,
#                       HSHEDS_TIF, HSHEDS_ADF, HSHEDS_AREA,
#                       RIVERS_AREA, RIVERS_TIF, RIVERS_FULL, RIVERS_ZIP)


Config.initialize()

class Image(ABC):
    # TODO: Ver de incluir area_envelope aca, ya que es comun a todas las
    #  clases

    @abstractmethod
    def prepare(self):
        raise NotImplementedError

    @abstractmethod
    def process(self):
        raise NotImplementedError


class SRTM(Image):

    def __init__(self):
        self.srtm_zip = Config.srtm('SRTM_ZIP')
        self.srtm_tif = Config.srtm('SRTM_TIF')
        self.area_polygon = Config.shapes('AREA_ENVELOPE')
        self.srtm_interest = Config.srtm('SRTM_AREA')
        self.fourier = Fourier(self.srtm_interest)
        self.groves = Groves()

    def prepare(self):
        unzip_resource(self.srtm_zip)
        resample_and_cut(self.srtm_tif, self.area_polygon, self.srtm_interest)
        return self

    def process(self):
        fourier_corrected = self.fourier.process()
        self.groves.prepare()
        return self.groves.process(fourier_corrected)


class Fourier(Image):

    def __init__(self, area_of_interest):
        self.aoi = area_of_interest

    def prepare(self):
        pass

    def process(self):
        srtm_raw = gdal.Open(self.aoi).ReadAsArray()
        return DetectApplyFourier().apply(srtm_raw)


class Groves(Image):

    def __init__(self):
        self.groves_class_zip = Config.groves('GROVES_ZIP')
        self.groves_class_tif = Config.groves('GROVES_TIF')
        self.area_polygon = Config.shapes('AREA_ENVELOPE')
        self.groves_interest = Config.groves('GROVES_AREA')

    def prepare(self):
        unzip_resource(self.groves_class_zip)
        resample_and_cut(self.groves_class_tif, self.area_polygon,
                         self.groves_interest)
        groves_class_raw = gdal.Open(self.groves_interest).ReadAsArray()
        return BinaryClosing(structure=np.ones((3, 3))).apply(groves_class_raw)

    def process(self, srtm):
        groves_class = self.prepare()
        return GrovesCorrectionsIter(groves_class).apply(srtm)


class HSHEDS(Image):

    def __init__(self):
        self.hsheds_zip = Config.hsheds('HSHEDS_ZIP')
        self.hsheds_adf = Config.hsheds('HSHEDS_ADF')
        self.hsheds_tif = Config.hsheds('HSHEDS_TIF')
        self.area_polygon = Config.shapes('AREA_ENVELOPE')
        self.hsheds_interest = Config.hsheds('HSHEDS_AREA')
        self.lagoons = Lagoons(self.hsheds_interest)
        self.rivers = Rivers()

    def prepare(self):
        unzip_resource(self.hsheds_zip)
        gdt_options = gdal.TranslateOptions(format='GTIFF')
        gdal.Translate(self.hsheds_tif, self.hsheds_adf, options=gdt_options)
        resample_and_cut(self.hsheds_tif, self.area_polygon,
                         self.hsheds_interest)
        return self

    def process(self):
        lagoons = self.lagoons.process()
        rivers = self.rivers.process(lagoons)
        return lagoons, rivers


class Lagoons(Image):

    def __init__(self, area_of_interest):
        self.aoi = area_of_interest

    def prepare(self):
        pass

    def process(self):
        hydro_sheds = gdal.Open(self.aoi).ReadAsArray()
        lagoons = LagoonsDetection()
        lagoons.apply(hydro_sheds)
        return lagoons


class Rivers(Image):
    def __init__(self):
        self.rivers_zip = Config.rivers('RIVERS_ZIP')
        self.rivers_full = Config.rivers('RIVERS_FULL')
        self.rivers_tif = Config.rivers('RIVERS_TIF')
        self.area_polygon = Config.shapes('AREA_ENVELOPE')
        self.rivers_interest = Config.rivers('RIVERS_AREA')

    def prepare(self):
        unzip_resource(self.rivers_zip)
        clip_lines_vector(self.rivers_full, self.area_polygon,
                          self.rivers_interest)
        rasterize_rivers(self.rivers_interest, self.rivers_tif)

    def process(self, lagoons):
        self.prepare()
        rivers = gdal.Open(self.rivers_tif).ReadAsArray()
        rivers_routed = ProcessRivers(lagoons.hsheds_nan_fixed).apply(rivers)
        return ClipLagoonsRivers(lagoons.mask_lagoons,
                                 rivers_routed).apply(rivers_routed)
