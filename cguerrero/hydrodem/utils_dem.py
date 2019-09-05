__author__ = "Cristian Guerrero Cordova"
__copyright__ = "Copyright 2017"
__credits__ = ["Cristian Guerrero Cordova"]
__version__ = "0.1"
__email__ = "cguerrerocordova@gmail.com"
__status__ = "Developing"

import copy
import os
import zipfile
from collections import Counter
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
from .sliding_window import (CircularWindow, SlidingWindow, NoCenterWindow,
                             IgnoreBorderInnerSliding)
from .filters import (MajorityFilter, BinaryErosion, ExpandFilter,
                      EnrouteRivers)



def array2raster(rasterfn, new_rasterfn, array):
    """
    Save image contained in an array format to a tiff georeferenced file.
    New image file will be located in new_rasterfn pathfile.
    Georeference is taken from rasterfn file.
    rasterfn and new_rasterfn must be strings with pathfile.
    """
    raster = gdal.Open(rasterfn)
    geo_transform = raster.GetGeoTransform()
    origin_x, pixel_width, _, origin_y, _, pixel_height = geo_transform

    cols = raster.RasterXSize
    rows = raster.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(new_rasterfn, cols, rows, 1, gdal.GDT_Float32)
    out_raster.SetGeoTransform((origin_x, pixel_width, 0, origin_y, 0,
                                pixel_height))
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(array)
    out_raster_srs = osr.SpatialReference()
    out_raster_srs.ImportFromWkt(raster.GetProjectionRef())
    out_raster.SetProjection(out_raster_srs.ExportToWkt())
    out_band.FlushCache()


def array2raster_simple(new_rasterfn, array):
    """
    Save an image 2-D array in a tiff file.
    :param new_rasterfn: path file target
    :param array: 2-D array
    :return: void
    """
    rows, cols = array.shape
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(new_rasterfn, cols, rows, 1, gdal.GDT_Float32)
    outband = out_raster.GetRasterBand(1)
    outband.WriteArray(array)
    outband.FlushCache()


def quadratic_filter(dem, window_size):
    """
    Smoothness filter: Apply a quadratic filter of smoothness
    :param dem: dem image
    """
    values = np.linspace(-window_size / 2 + 1, window_size / 2, window_size)
    xx, yy = np.meshgrid(values, values)
    r0 = window_size ** 2
    r1 = (xx * xx).sum()
    r2 = (xx * xx * xx * xx).sum()
    r3 = (xx * xx * yy * yy).sum()

    dem_sliding = SlidingWindow(dem, window_size=window_size)
    smoothed = dem.copy()

    for window, center in dem_sliding:
        s1 = window.sum()
        s2 = (window * xx * xx).sum()
        s3 = (window * yy * yy).sum()
        smoothed[center] = ((s2 + s3) * r1 - s1 * (r2 + r3)) / \
                           (2 * r1 ** 2 - r0 * (r2 + r3))
    return smoothed

def correct_nan_values(dem):
    """
    Correct values lower than zero, generally with extremely lowest values.
    """
    mask_nan = (dem < 0.0) * 1
    sliding_nans = SlidingWindow(mask_nan, window_size=3, iter_over_ones=True)
    dem_sliding = NoCenterWindow(dem, window_size=3)
    for _, center in sliding_nans:
        neighbours_of_nan = dem_sliding[center].ravel()
        neighbours_positives = \
            list(filter(lambda x: x >= 0, neighbours_of_nan))
        dem[center] = sum(neighbours_positives) / len(neighbours_positives)
    return dem


def filter_isolated_pixels(image_to_filter, window_size):
    """
    Remove isolated pixels detected to be part of a mask.
    """
    sliding = NoCenterWindow(image_to_filter, window_size=window_size,
                             iter_over_ones=True)
    for window, center in sliding:
        image_to_filter[center] = 1. if np.any(window > 0) else 0.
    return image_to_filter

def filter_blanks(image_to_filter, window_size):
    """
    Define the filter to detect blanks in a fourier transform image.
    """

    filtered_image = np.zeros(image_to_filter.shape)
    fourier_transform = IgnoreBorderInnerSliding(image_to_filter,
                                                 window_size=window_size,
                                                 inner_size=5)
    for window, center in fourier_transform:
        mean_neighbor = np.nanmean(window)
        real_center = center[0] - window_size // 2, \
                      center[1] - window_size // 2,
        if image_to_filter[real_center] > (4 * mean_neighbor):
            filtered_image[real_center] = 1.0
    image_modified = image_to_filter * (1 - filtered_image)
    return filtered_image, image_modified

def get_mask_fourier(quarter_fourier):
    """
    Perform iterations of filter blanks functions and produce a final mask
    with blanks of fourier transform.
    :param quarter_fourier: fourier transform image.
    :return: mask with detected blanks
    """
    final_mask_image = detect_blanks_fourier(quarter_fourier)
    final_mask_without_pts = filter_isolated_pixels(final_mask_image, 3)
    return ExpandFilter(window_size=13).apply(final_mask_without_pts)
    # return expand_filter(final_mask_without_pts, 13)


def detect_blanks_fourier(quarter_fourier):
    """
    Parameters
    ----------
    quarter_fourier

    Returns
    -------

    """
    final_mask_image = np.zeros(quarter_fourier.shape)
    for i in (0, 1):
        filtered_blanks, quarter_fourier = filter_blanks(quarter_fourier, 55)
        final_mask_image += filtered_blanks
    return final_mask_image




def detect_apply_fourier(image_to_correct):
    """
    Detect blanks in Fourier transform image, create mask and apply fourier.
    """
    # TODO: Aca se abre el archivo dentro de la funcion, ver si hacerlo fuera
    image_to_correct = gdal.Open(image_to_correct).ReadAsArray()
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
    first_quarter_mask = get_mask_fourier(fst_quarter_fourier)
    second_quarter_mask = get_mask_fourier(snd_quarter_fourier)
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

def process_srtm(srtm_fourier, tree_class_file):
    """
    Perform the processing corresponding to SRTM file.
    """
    srtm_fourier_sua = quadratic_filter(srtm_fourier, 15)
    dem_highlighted = srtm_fourier - srtm_fourier_sua
    mask_height_greater_than_15_mt = (dem_highlighted > 1.5) * 1.0
    # TODO: Uniformizar, input, uno entrada otro path
    tree_class_raw = gdal.Open(tree_class_file).ReadAsArray()
    tree_class = ndimage.binary_closing(tree_class_raw, np.ones((3, 3)))
    tree_class_height_15_mt = tree_class * mask_height_greater_than_15_mt
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

def get_lagoons_hsheds(hsheds_input):
    # majority_filter = MajorityFilter(window_size=11)
    hsheds_maj_filter11 = MajorityFilter(window_size=11).apply(hsheds_input)
    hsheds_maj_filter11_ero2 = \
        BinaryErosion(iterations=2).apply(hsheds_maj_filter11)
    # hsheds_maj_filter11_ero2 = ndimage.binary_erosion(hsheds_maj_filter11,
    #                                                   iterations=2)
    hsheds_maj_filter11_ero2_expand7 = \
        ExpandFilter(window_size=7).apply(hsheds_maj_filter11_ero2)
    hsheds_maj11_ero2_expand7_prod_maj11 = \
        hsheds_maj_filter11_ero2_expand7 * hsheds_maj_filter11
    hsheds_mask_lagoons_values = ndimage.grey_dilation(
        hsheds_maj11_ero2_expand7_prod_maj11, size=(7, 7))
    return hsheds_mask_lagoons_values


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
    mask_canyons = (rivers > 0) * 1
    mask_canyons_expanded3 = ExpandFilter(window_size=3).apply(mask_canyons)
    rivers_routed = route_rivers(hsheds_area_interest, mask_canyons_expanded3,
                                 3)
    rivers_routed_closing = binary_closing(rivers_routed)
    intersection_lag_can = rivers_routed_closing * hsheds_mask_lagoons
    intersection_lag_can_mask = (intersection_lag_can > 0) * 1
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
