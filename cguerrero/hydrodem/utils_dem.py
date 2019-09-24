__author__ = "Cristian Guerrero Cordova"
__copyright__ = "Copyright 2019"
__credits__ = ["Cristian Guerrero Cordova"]
__version__ = "0.1"
__email__ = "cguerrerocordova@gmail.com"
__status__ = "Developing"

import os
import zipfile

import ogr
import osr
from osgeo import gdal

TEMP_REPROJECTED_TO_CUT = "temp_reprojected.tif"


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


# def process_srtm(srtm_fourier, tree_classification):
#     """
#     Perform the processing corresponding to SRTM file.
#     """
#
#     tree_class = BinaryClosing(structure=np.ones((3, 3))).apply(tree_classification)
#
#     # srtm_fourier_sua = QuadraticFilter(window_size=15).apply(srtm_fourier)
#     # dem_highlighted = SubtractionFilter(minuend=srtm_fourier).apply(srtm_fourier_sua)
#     # mask_tall_trees = MaskTallGroves().apply(dem_highlighted)
#     # tree_height_15_mt = ProductFilter(factor=tree_class).apply(mask_tall_trees)
#     # tree_height_15_mt_compl = SubtractionFilter(minuend=1).apply(tree_height_15_mt)
#     #
#     #
#     # trees_removed = dem_highlighted * tree_height_15_mt_compl
#     # dem_corrected_15 = srtm_fourier_sua + trees_removed
#     # return dem_corrected_15
#
#     return GrovesCorrection(tree_class).apply(srtm_fourier)



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


def shape_enveloping(shape_area_input, shape_over_area):
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


# def process_rivers(hsheds, mask_lagoons, rivers):
#     """
#     Get rivers from hsheds DEM
#     :param hsheds: DEM HSHEDS to get rivers
#     :param mask_lagoons: Mask Lagoons to exclude from rivers
#     :return: Rivers detected
#     """
#     rivers_routed = ProcessRivers(hsheds).apply(rivers)
#     return ClipLagoonsRivers(mask_lagoons, rivers_routed).apply(rivers_routed)


def rasterize_rivers(rivers_shape, rivers_tif):

    pixel_size = 90
    layer = os.path.splitext(os.path.basename(rivers_tif))[0]
    gdr_options = gdal.RasterizeOptions(attribute='UP_CELLS',
                                        yRes=pixel_size, xRes=pixel_size,
                                        outputType=gdal.GDT_Float32,
                                        layers=layer)
    # layers='rivers_area_interest')
    gdal.Rasterize(rivers_tif, rivers_shape, options=gdr_options)


def unzip_resource(zip_file):
    """
    Uncompress zip file
    :param zip_file: path where file to uncompress is located
    """
    dir_name = os.path.dirname(zip_file)
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall(dir_name)
    zip_ref.close()

# def clean_workspace():
#     to_clean = [TREE_CLASS_AREA, SRTM_AREA_OVER, SRTM_FILE_TIF,
#                 HSHEDS_AREA_OVER, RIVERS_TIF, HSHEDS_FILE_TIFF,
#                 TREE_CLASS_INPUT]
#     for file in to_clean:
#         try:
#             os.remove(file)
#         except OSError:
#             pass
