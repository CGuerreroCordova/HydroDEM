__author__ = "Cristian Guerrero Cordova"
__copyright__ = "Copyright 2017"
__credits__ = ["Cristian Guerrero Cordova"]
__version__ = "0.1"
__email__ = "cguerrerocordova@gmail.com"
__status__ = "Developing"

import os
import zipfile
import numpy as np
import ogr
import osr
from osgeo import gdal
from scipy import fftpack
from scipy import ndimage
from scipy.ndimage.morphology import binary_closing

from .settings import (RIVERS_TIF, TEMP_REPROJECTED_TO_CUT, TREE_CLASS_AREA,
                       SRTM_AREA_INTEREST_OVER, HSHEDS_AREA_INTEREST_OVER,
                       HSHEDS_FILE_TIFF, SRTM_FILE_INPUT, TREE_CLASS_INPUT)
from .filters import (ExpandFilter, EnrouteRivers, QuadraticFilter,
                      MaskPositives, MaskFourier, SubtractionFilter)


def array2raster(new_rasterfn, array, rasterfn=None):
    """
    Save image contained in an array format to a tiff georeferenced file.
    New image file will be located in new_rasterfn pathfile.
    Georeference is taken from rasterfn file.
    rasterfn and new_rasterfn must be strings with pathfile.
    """

    rows, cols = array.shape
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(new_rasterfn, cols, rows, 1, gdal.GDT_Float32)

    if rasterfn:
        raster = gdal.Open(rasterfn)
        geo_transform = raster.GetGeoTransform()
        origin_x, pixel_width, _, origin_y, _, pixel_height = geo_transform
        out_raster.SetGeoTransform((origin_x, pixel_width, 0, origin_y, 0,
                                    pixel_height))
        out_raster_srs = osr.SpatialReference()
        out_raster_srs.ImportFromWkt(raster.GetProjectionRef())
        out_raster.SetProjection(out_raster_srs.ExportToWkt())

    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.FlushCache()


def detect_apply_fourier(image_to_correct):
    """
    Detect blanks in Fourier transform image, create mask and apply fourier.
    """
    fourier_transform = fftpack.fft2(image_to_correct)
    fourier_transform_shifted = fftpack.fftshift(fourier_transform)
    fft_transform_abs = np.abs(fourier_transform_shifted)
    ny, nx = fft_transform_abs.shape
    x_odd = nx & 1
    y_odd = ny & 1
    middle_x = int(nx // 2)
    middle_y = int(ny // 2)
    margin = 10
    fst_quarter_fourier = fft_transform_abs[:middle_y - margin, :middle_x -
                                                                 margin]
    snd_quarter_fourier = fft_transform_abs[:middle_y - margin, middle_x +
                                                                margin +
                                                                x_odd:nx]
    # p = Pool(2)
    # masks_fourier = p.map(get_mask_fourier, [fst_quarter_fourier,
    #                                          snd_quarter_fourier])
    # first_quarter_mask = masks_fourier[0]
    # second_quarter_mask = masks_fourier[1]}
    first_quarter_mask = MaskFourier().apply(fst_quarter_fourier)
    second_quarter_mask = MaskFourier().apply(snd_quarter_fourier)
    fst_complete_quarter = np.zeros((middle_y, middle_x))
    snd_complete_quarter = np.zeros((middle_y, middle_x))
    fst_complete_quarter[0:middle_y - margin, 0:middle_x - margin] = \
        first_quarter_mask
    snd_complete_quarter[0:middle_y - margin, margin:middle_x] = \
        second_quarter_mask
    reverse_x = (middle_x - 1) - np.arange(0, middle_x)
    reverse_y = (middle_y - 1) - np.arange(0, middle_y)
    indices = np.ix_(reverse_y, reverse_x)
    fth_complete_quarter = fst_complete_quarter[indices]
    trd_complete_quarter = snd_complete_quarter[indices]

    masks_fourier = np.zeros((2 * middle_y + y_odd, 2 * middle_x + x_odd))
    masks_fourier[0:middle_y, 0:middle_x] = fst_complete_quarter
    masks_fourier[0:middle_y, middle_x + x_odd:nx] = snd_complete_quarter
    masks_fourier[middle_y + y_odd:ny, 0:middle_x] = trd_complete_quarter
    masks_fourier[middle_y + y_odd:ny, middle_x + x_odd:nx] = \
        fth_complete_quarter
    complement_mask_fourier = 1 - masks_fourier
    fourier_transform_corrected = fourier_transform_shifted \
                                  * complement_mask_fourier
    shifted = fftpack.ifftshift(fourier_transform_corrected)
    corrected_image = fftpack.ifft2(shifted)
    return corrected_image


def process_srtm(srtm_fourier, tree_classification):
    """
    Perform the processing corresponding to SRTM file.
    """
    srtm_fourier_sua = QuadraticFilter(window_size=15).apply(srtm_fourier)
    dem_highlighted = SubtractionFilter(minuend=srtm_fourier).apply(
        srtm_fourier_sua)
    mask_tall_trees = (dem_highlighted > 1.5) * 1.0
    tree_class = ndimage.binary_closing(tree_classification, np.ones((3, 3)))
    tree_class_height_15_mt = tree_class * mask_tall_trees
    tree_class_height_15_mt_compl = 1 - tree_class_height_15_mt
    trees_removed = dem_highlighted * tree_class_height_15_mt_compl
    dem_corrected_15 = trees_removed + srtm_fourier_sua
    return dem_corrected_15


def resample_and_cut(orig_image, shape_file, target_path):
    """
    Resample the DEM to corresponding projection.
    Cut the area defined by shape
    """

    pixel_size = 90
    in_file = gdal.Open(orig_image)
    gdal.Warp(TEMP_REPROJECTED_TO_CUT, in_file, dstSRS='EPSG:22184')
    gdw_options = gdal.WarpOptions(cutlineDSName=shape_file,
                                   cropToCutline=True,
                                   xRes=pixel_size, yRes=pixel_size,
                                   outputType=gdal.GDT_Float32)
    gdal.Warp(target_path, TEMP_REPROJECTED_TO_CUT, dstSRS='EPSG:22184',
              options=gdw_options)
    try:
        os.remove(TEMP_REPROJECTED_TO_CUT)
    except OSError:
        pass


def get_shape_over_area(shape_area_input, shape_over_area):
    """
    Create a rectangular shape file covering all area of interest and
    parallel to the equator.
    """
    area_driver = ogr.GetDriverByName("ESRI Shapefile")
    shapefile = area_driver.Open(shape_area_input, 0)
    layer = shapefile.GetLayer()
    feature = layer.GetFeature(0)
    geometry = feature.GetGeometryRef()
    minLong, maxLong, minLat, maxLat = geometry.GetEnvelope()
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(minLong, maxLat)
    ring.AddPoint(maxLong, maxLat)
    ring.AddPoint(maxLong, minLat)
    ring.AddPoint(minLong, minLat)
    ring.AddPoint(minLong, maxLat)

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    # Create the output Layer
    outDriver = ogr.GetDriverByName("ESRI Shapefile")
    srs = osr.SpatialReference()
    srs.ImportFromWkt(layer.GetSpatialRef().ExportToWkt())

    if os.path.exists(shape_over_area):
        outDriver.DeleteDataSource(shape_over_area)

    outDataSource = outDriver.CreateDataSource(shape_over_area)
    outLayer = outDataSource.CreateLayer("shape_over_area", srs,
                                         geom_type=ogr.wkbPolygon)

    idField = ogr.FieldDefn("id", ogr.OFTInteger)
    outLayer.CreateField(idField)

    # Create the feature and set values
    featureDefn = outLayer.GetLayerDefn()
    feature = ogr.Feature(featureDefn)
    feature.SetGeometry(poly)
    feature.SetField("id", 1)
    outLayer.CreateFeature(feature)

    # Save and close DataSource
    outDataSource.Destroy()


def clip_lines_vector(lines_vector, polygon_vector, lines_output):
    """
    Clip a lines vector file using polygon vector file
    :param lines_vector: lines vector file to clip
    :param polygon_vector: polygon to use as a clipper
    :param lines_output: line vector file clipped
    """
    rivers_driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource_rivers = rivers_driver.Open(lines_vector, 0)
    rivers_layer = dataSource_rivers.GetLayer()

    area_driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource_area = area_driver.Open(polygon_vector, 0)
    area_layer = dataSource_area.GetLayer()

    # Create the output Layer
    outDriver = ogr.GetDriverByName("ESRI Shapefile")

    srs = osr.SpatialReference()
    srs.ImportFromWkt(rivers_layer.GetSpatialRef().ExportToWkt())

    # Remove output shapefile if it already exists
    if os.path.exists(lines_output):
        outDriver.DeleteDataSource(lines_output)
    # Create the output shapefile
    outDataSource = outDriver.CreateDataSource(lines_output)
    outLayer = outDataSource.CreateLayer("rivers_clipped", srs,
                                         geom_type=ogr.wkbLineString)
    # Clip
    rivers_layer.Clip(area_layer, outLayer)

    # Close DataSources
    dataSource_rivers.Destroy()
    dataSource_area.Destroy()
    outDataSource.Destroy()


def process_rivers(hsheds_area_interest, hsheds_mask_lagoons, rivers_shape):
    """
    Get rivers from hsheds DEM
    :param hsheds_area_interest: DEM HSHEDS to get rivers
    :param hsheds_mask_lagoons: Mask Lagoons to exclude from rivers
    :return: Rivers detected
    """
    pixel_size = 90
    gdr_options = gdal.RasterizeOptions(attribute='UP_CELLS',
                                        yRes=pixel_size, xRes=pixel_size,
                                        outputType=gdal.GDT_Float32,
                                        layers='rivers_area_interest')
    gdal.Rasterize(RIVERS_TIF, rivers_shape, options=gdr_options)
    rivers = gdal.Open(RIVERS_TIF).ReadAsArray()
    mask_rivers = MaskPositives().apply(rivers)
    mask_rivers_expanded3 = ExpandFilter(window_size=3).apply(mask_rivers)
    rivers_routed = EnrouteRivers(window_size=3).apply(hsheds_area_interest,
                                                       mask_rivers_expanded3)
    rivers_routed_closing = binary_closing(rivers_routed)
    intersection_lag_can = rivers_routed_closing * hsheds_mask_lagoons
    intersection_lag_can_mask = MaskPositives().apply(intersection_lag_can)
    rivers_routed_closing = np.bitwise_xor(rivers_routed_closing,
                                           intersection_lag_can_mask)
    return rivers_routed_closing

def uncompress_zip_file(zip_file):
    """
    Uncompress zip file
    :param zip_file: path where file to uncompress is located
    """
    dir_name = os.path.dirname(zip_file)
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall(dir_name)
    zip_ref.close()


def clean_workspace():
    to_clean = [TREE_CLASS_AREA, SRTM_AREA_INTEREST_OVER, SRTM_FILE_INPUT,
                HSHEDS_AREA_INTEREST_OVER, RIVERS_TIF, HSHEDS_FILE_TIFF,
                TREE_CLASS_INPUT]
    for file in to_clean:
        try:
            os.remove(file)
        except OSError:
            pass
