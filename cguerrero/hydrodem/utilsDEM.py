__author__ = "Cristian Guerrero Cordova"
__copyright__ = "Copyright 2017"
__credits__ = ["Cristian Guerrero Cordova"]
__version__ = "0.1"
__email__ = "cguerrerocordova@gmail.com"
__status__ = "Developing"

import os
import subprocess
from collections import Counter
from multiprocessing import Pool
import glob

import datetime
import numpy as np
import ogr, osr
from osgeo import gdal
from scipy import fftpack
from scipy import ndimage
import copy
from scipy.ndimage.morphology import binary_closing
import zipfile

from settings import (RIVERS_TIF, TEMP_REPROJECTED_TO_CUT, TREE_CLASS_AREA,
                      SRTM_AREA_INTEREST_OVER, HSHEDS_AREA_INTEREST_OVER)

def array2raster(rasterfn, new_rasterfn, array):
    """
    Save image contained in an array format to a tiff georeferenced file.
    New image file will be located in new_rasterfn pathfile.
    Georeference is taken from rasterfn file.
    rasterfn and new_rasterfn must be strings with pathfile.
    """
    raster = gdal.Open(rasterfn)
    geo_transform = raster.GetGeoTransform()
    origin_x = geo_transform[0]
    origin_y = geo_transform[3]
    pixel_width = geo_transform[1]
    pixel_height = geo_transform[5]
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
    cols = array.shape[1]
    rows = array.shape[0]
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(new_rasterfn, cols, rows, 1, gdal.GDT_Float32)
    outband = out_raster.GetRasterBand(1)
    outband.WriteArray(array)
    outband.FlushCache()

def majority_filter(image_to_filter, window_size):
    """
    The value assigned to the center will be the most frequent value,
    contained in a windows of window_size x window_size
    """
    ny = image_to_filter.shape[0]
    nx = image_to_filter.shape[1]
    filtered_image = np.zeros((ny, nx))
    right_up_side = window_size / 2
    left_down_side = window_size / 2 + 1
    for j in range(right_up_side, ny - left_down_side):
        for i in range(left_down_side, nx - right_up_side):
            vertical_kernel = image_to_filter[
                              j - right_up_side: j + left_down_side,
                              i - (right_up_side - 1): i + (left_down_side - 1)
                              ]
            horizontal_kernel = image_to_filter[
                                j - (right_up_side - 1): j + (
                                        left_down_side - 1),
                                i - right_up_side: i + left_down_side
                                ]
            c = Counter(vertical_kernel.flatten() +
                        horizontal_kernel.flatten())
            value, count = c.most_common()[0]
            if count >= (((window_size * window_size) - 1) / 2):
                filtered_image[j, i] = value / 2
    return filtered_image


def expand_filter(image_to_expand, window_size):
    """
    The value assigned to the center will be 1 if at least one pixel
    inside the kernel is 1
    """
    ny = image_to_expand.shape[0]
    nx = image_to_expand.shape[1]
    expanded_image = np.zeros((ny, nx))
    right_up_side = window_size / 2
    left_down_side = window_size / 2 + 1
    s = {0}
    for j in range(right_up_side, ny - left_down_side):
        for i in range(left_down_side, nx - right_up_side):
            vertical_kernel = image_to_expand[
                              j - right_up_side: j + left_down_side,
                              i - (right_up_side - 1): i + (left_down_side - 1)
                              ]
            horizontal_kernel = image_to_expand[
                                j - (right_up_side - 1): j + (
                                        left_down_side - 1),
                                i - right_up_side: i + left_down_side
                                ]
            v = set(vertical_kernel.flatten())
            h = set(horizontal_kernel.flatten())
            t = v | h
            if len(t - s) > 0:
                expanded_image[j, i] = 1
    return expanded_image


def route_rivers(dem_in, maskRivers, window_size):
    """
    Apply homogeneity to canyons. Specific defined for images with flow stream.
    """
    dem = copy.deepcopy(dem_in)
    ny = dem.shape[0]
    nx = dem.shape[1]
    indices = np.nonzero(maskRivers > 0.0)
    rivers_enrouted = np.zeros((ny, nx))
    right_up_side = window_size / 2
    left_down_side = window_size / 2 + 1
    neighbors = np.zeros((window_size, window_size))
    for x in range(len(indices[0])):
        j = indices[0][x]
        i = indices[1][x]
        if (left_down_side < i < nx - (right_up_side - 1) and
                left_down_side < j < ny - (right_up_side - 1)):
            neighbors = dem[j - (window_size - 2): j + (window_size - 1),
                        i - (window_size - 2): i + (window_size - 1)]
            neighbors_flat = neighbors.flatten()
            neighbor_min = np.amin(neighbors_flat)
            indices_min = np.nonzero(neighbors == neighbor_min)
            for z in range(len(indices_min[0])):
                min_j = indices_min[0][z]
                min_i = indices_min[1][z]
                min_j_index = j - (window_size - 2) + min_j
                min_i_index = i - (window_size - 2) + min_i
                rivers_enrouted[min_j_index, min_i_index] = 1
                dem[min_j_index, min_i_index] = 10000
    return rivers_enrouted


def quadratic_filter(z):
    """
    Smoothness filter: Apply a quadratic filter of smoothness
    :param z: dem image
    """
    n = 15
    xx, yy = np.meshgrid(np.linspace(-n / 2 + 1, n / 2, n),
                         np.linspace(-n / 2 + 1, n / 2, n))
    r0 = n ** 2
    r1 = (xx * xx).sum()
    r2 = (xx * xx * xx * xx).sum()
    r3 = (xx * xx * yy * yy).sum()
    ny = z.shape[0]
    nx = z.shape[1]
    zp = z.copy()
    for i in range(n / 2, nx - n / 2):
        for j in range(n / 2, ny - n / 2):
            z_aux = z[j - n / 2:j + n / 2 + 1, i - n / 2:i + n / 2 + 1]
            s1 = z_aux.sum()
            s2 = (z_aux * xx * xx).sum()
            s3 = (z_aux * yy * yy).sum()
            zp[j, i] = ((s2 + s3) * r1 - s1 * (r2 + r3)) / (
                    2 * r1 ** 2 - r0 * (r2 + r3))
    dem_quadratic = zp
    return dem_quadratic

def correct_nan_values(dem):
    """
    Correct values lower than zero, generally with extremely lowest values.
    """
    ny = dem.shape[0]
    nx = dem.shape[1]
    indices = np.nonzero(dem < 0.0)
    dem_corrected_nan = dem.copy()
    for x in range(len(indices[0])):
        j = indices[0][x]
        i = indices[1][x]
        if i != 0 and j != 0 and i != nx - 1 and j != ny - 1:
            neighbors = dem[j - 1, i - 1: i + 2].flatten().tolist() \
                        + dem[j + 1, i - 1: i + 2].flatten().tolist() \
                        + [dem[j, i - 1]] + [dem[j, i + 1]]
            neighbors_array = np.array(neighbors)
            greater_than_zero = (neighbors_array > 0) * 1
            c = Counter(greater_than_zero.flatten())
            value, count = c.most_common()[0]
            n = ((value == 0) * 1)
            sum_neighbors = sum(neighbors * np.absolute(n - greater_than_zero))
            mean_neighbors = sum_neighbors / count
            dem_corrected_nan[j, i] = mean_neighbors
    return dem_corrected_nan


def filter_isolated_pixels(image_to_filter, window_size):
    """
    Remove isolated pixels detected to be part of a mask.
    """
    ny = image_to_filter.shape[0]
    nx = image_to_filter.shape[1]
    filtered_image = np.zeros((ny, nx))
    right_up_side = window_size / 2
    left_down_side = window_size / 2 + 1
    margin = 0
    for j in range(right_up_side, ny - left_down_side):
        for i in range(left_down_side, nx - right_up_side):
            if image_to_filter[j, i] == 1:
                above_neighbors = image_to_filter[
                                  j - right_up_side: j - margin,
                                  i - right_up_side: i + left_down_side
                                  ].flatten().tolist()
                below_neighbors = image_to_filter[
                                  j + 1 + margin: j + left_down_side,
                                  i - right_up_side: i + left_down_side
                                  ].flatten().tolist()
                right_neighbors = image_to_filter[
                                  j, i - right_up_side: i - margin
                                  ].flatten().tolist()
                left_neighbors = image_to_filter[
                                 j, i + 1 + margin: i + left_down_side
                                 ].flatten().tolist()
                neighbors = above_neighbors + below_neighbors \
                            + right_neighbors + left_neighbors
                sum_neighbors = sum(neighbors)
                filtered_image[j, i] = (sum_neighbors > 0) * 1
    return filtered_image


def filter_blanks(image_to_filter, window_size):
    """
    Define the filter to detect blanks in a fourier transform image.
    """
    ny = image_to_filter.shape[0]
    nx = image_to_filter.shape[1]
    filtered_image = np.zeros((ny, nx))
    right_up_side = window_size / 2
    left_down_side = window_size / 2 + 1
    margin = 5
    for j in range(0, ny):
        for i in range(0, nx):
            if j < right_up_side:
                above_neighbors = image_to_filter[
                                  0: j,
                                  i - right_up_side: i + left_down_side
                                  ].flatten().tolist()
            else:
                above_neighbors = image_to_filter[
                                  j - right_up_side: j - margin,
                                  i - right_up_side: i + left_down_side
                                  ].flatten().tolist()
            if j > (ny - left_down_side):
                below_neighbors = image_to_filter[
                                  j + 1: ny,
                                  i - right_up_side:i + left_down_side
                                  ].flatten().tolist()
            else:
                below_neighbors = image_to_filter[
                                  j + 1 + margin:j + left_down_side,
                                  i - right_up_side:i + left_down_side
                                  ].flatten().tolist()
            if i > (nx - right_up_side):
                right_neighbors = image_to_filter[
                                  j, i + 1:nx
                                  ].flatten().tolist()
            else:
                right_neighbors = image_to_filter[
                                  j, i + 1 + margin:i + left_down_side
                                  ].flatten().tolist()
            if i < right_up_side:
                left_neighbors = image_to_filter[j, 0:i].flatten().tolist()
            else:
                left_neighbors = image_to_filter[
                                 j, i - right_up_side:i - margin
                                 ].flatten().tolist()
            neighbors = above_neighbors + below_neighbors + right_neighbors \
                        + left_neighbors
            mean_neighbor = sum(neighbors) / len(neighbors)
            if image_to_filter[j, i] > (4 * mean_neighbor):
                filtered_image[j, i] = mean_neighbor
    mask_not_filtered = (filtered_image == 0) * 1.0
    mask_filtered = 1 - mask_not_filtered
    image_modified = image_to_filter * mask_not_filtered
    return mask_filtered, image_modified


def get_mask_fourier(quarter_fourier):
    """
    Perform iterations of filter blanks functions and produce a final mask
    with blanks of fourier transform.
    :param quarter_fourier: fourier transform image.
    :return: mask with detected blanks
    """
    quarter_ny = quarter_fourier.shape[0]
    quarter_nx = quarter_fourier.shape[1]
    final_mask_image = np.zeros((quarter_ny, quarter_nx))
    # TODO: Eliminar variables innecesarias
    image_modified = np.zeros((quarter_ny, quarter_nx))
    for i in range(0, 2):
        (filtered_blank_image, image_modified) = \
            filter_blanks(quarter_fourier, 55)
        final_mask_image = final_mask_image + filtered_blank_image
        quarter_fourier = image_modified
    final_mask_im_without_pts = filter_isolated_pixels(final_mask_image, 3)
    final_mask = expand_filter(final_mask_im_without_pts, 13)
    return final_mask


def detect_apply_fourier(image_to_correct):
    """
    Detect blanks in Fourier transform image, create mask and apply fourier.
    """
    image_to_correct_array = gdal.Open(image_to_correct).ReadAsArray()
    fourier_transform = fftpack.fft2(image_to_correct_array)
    fourier_transform_shifted = fftpack.fftshift(fourier_transform)
    fft_transform_abs = np.abs(fourier_transform_shifted)
    nx = fft_transform_abs.shape[1]
    ny = fft_transform_abs.shape[0]
    x_odd = nx & 1
    y_odd = ny & 1
    middle_x = nx / 2
    middle_y = ny / 2
    margin = 10
    fst_quarter_fourier = fft_transform_abs[
                          0:middle_y - margin,
                          0:middle_x - margin
                          ]
    snd_quarter_fourier = fft_transform_abs[
                          0:middle_y - margin,
                          middle_x + margin + x_odd:nx
                          ]
    p = Pool(2)
    masks_fourier = p.map(get_mask_fourier, [fst_quarter_fourier,
                                             snd_quarter_fourier])
    first_quarter_mask = masks_fourier[0]
    second_quarter_mask = masks_fourier[1]
    fst_complete_quarter = np.zeros((middle_y, middle_x))
    snd_complete_quarter = np.zeros((middle_y, middle_x))
    fst_complete_quarter[0:middle_y - margin, 0:middle_x - margin] = \
        first_quarter_mask
    snd_complete_quarter[0:middle_y - margin, margin:middle_x] = \
        second_quarter_mask
    reverse_array_x = (middle_x - 1) - np.arange(0, middle_x)
    reverse_array_y = (middle_y - 1) - np.arange(0, middle_y)
    indices = np.ix_(reverse_array_y, reverse_array_x)
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
    srtm_fourier_sua = quadratic_filter(srtm_fourier)
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
    gdal_warp = "gdalwarp.exe"
    proj = '"+proj=tmerc +lat_0=-90 +lon_0=-63 +k=1 +x_0=4500000 +y_0=0 ' \
           '+ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"'
    pixel_size = 90
    command_line = gdal_warp + ' -t_srs ' + proj + ' ' + orig_image + ' ' \
                   + TEMP_REPROJECTED_TO_CUT
    os.system(command_line)
    command_line = gdal_warp + " -cutline " + shape_file + " -crop_to_cutline " + \
                   TEMP_REPROJECTED_TO_CUT + " " + target_path + " -tr " + \
                   str(pixel_size) + " " + str(pixel_size) + " -ot Float32"
    os.system(command_line)
    try:
        os.remove(TEMP_REPROJECTED_TO_CUT)
    except OSError:
        pass


def calling_system_call(command_line):
    """
    Execute command line in operative system call
    :param command_line: sentence to execute
    """
    command_line_list = command_line.split()
    # FNULL = open(os.devnull, 'w')
    # subprocess.call(command_line_list, stdout=FNULL, stderr=subprocess.STDOUT)
    subprocess.call(command_line_list)


def get_shape_over_area(image_temp, shape_over_area):
    """
    Return a rectangular shape file covering all area of interest and
    parallel to the equator.
    """
    shape_over_without_extension = os.path.splitext(shape_over_area)[0]
    file_list = glob.glob(shape_over_without_extension + ".*")
    for file_path in file_list:
        try:
            os.remove(file_path)
        except:
            print("Error while deleting file : ", file_path)

    command = 'gdaltindex ' + shape_over_area + ' ' + image_temp
    calling_system_call(command)

def get_lagoons_hsheds(hsheds_input):
    hsheds_maj_filter11 = majority_filter(hsheds_input, 11)
    hsheds_maj_filter11_ero2 = ndimage.binary_erosion(hsheds_maj_filter11,
                                                      iterations=2)
    hsheds_maj_filter11_ero2_expand7 = expand_filter(hsheds_maj_filter11_ero2,
                                                     7)
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
    cols = hsheds_mask_lagoons.shape[1]
    rows = hsheds_mask_lagoons.shape[0]
    # Convert to Raster Rivers
    command_line = 'gdal_rasterize -a UP_CELLS -ts ' + str(cols) + ' ' + \
                   str(rows) + ' -l rivers_area_interest ' + rivers_shape + \
                   ' ' + RIVERS_TIF
    calling_system_call(command_line)

    river_array = gdal.Open(RIVERS_TIF).ReadAsArray()
    mask_canyons_array = (river_array > 0) * 1
    mask_canyons_expanded3 = expand_filter(mask_canyons_array, 3)
    rivers_routed = route_rivers(hsheds_area_interest, mask_canyons_expanded3,
                                 3)
    rivers_routed_closing = binary_closing(rivers_routed)
    intersection_lag_can = rivers_routed_closing * hsheds_mask_lagoons
    intersection_lag_can_mask = (intersection_lag_can > 0)
    rivers_routed_closing -= intersection_lag_can_mask

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
    to_clean = [TREE_CLASS_AREA, SRTM_AREA_INTEREST_OVER,
                HSHEDS_AREA_INTEREST_OVER]
    for file in to_clean:
        try:
            os.remove(file)
        except OSError:
            pass
