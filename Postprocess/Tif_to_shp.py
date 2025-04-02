from osgeo import gdal, ogr, osr
import os
import time

os.environ['PROJ_LIB'] = r'D:\ProgramData\anaconda3\envs\torch10_wwr\Lib\site-packages\osgeo\data\proj'

def RasterToPolygon_conversion(raster, outshp):
    inraster = gdal.Open(raster)  # 读取路径中的栅格数据
    inband = inraster.GetRasterBand(1)  # 这个波段就是最后想要转为矢量的波段，如果是单波段数据的话那就都是1
    prj = osr.SpatialReference()
    prj.ImportFromWkt(inraster.GetProjection())  # 读取栅格数据的投影信息，用来为后面生成的矢量做准备

    drv = ogr.GetDriverByName("ESRI Shapefile")

    Polygon = drv.CreateDataSource(outshp)  # 创建一个目标文件
    Poly_layer = Polygon.CreateLayer(raster[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon)  # 对shp文件创建一个图层，定义为多个面类
    newField = ogr.FieldDefn('value', ogr.OFTInteger)  # 给目标shp文件添加一个字段，用来存储原始栅格的pixel value
    Poly_layer.CreateField(newField)

    gdal.FPolygonize(inband, None, Poly_layer, 0)  # 核心函数，执行的就是栅格转矢量操作
    Polygon.SyncToDisk()
    Polygon = None




if __name__ == '__main__':
    root = r'Y:\Training_Data\image\province_pred'
    os.makedirs(os.path.join(root, 'shp'),exist_ok=True)
    tif_list = os.listdir(root)
    tif_list.reverse()
    for tif_name in tif_list:
        print(os.path.join(root, tif_name))
        if not os.path.exists(os.path.join(root, 'shp', tif_name).replace('.tif', '.shp')):
            RasterToPolygon_conversion(os.path.join(root, tif_name),
                                   os.path.join(root, 'shp', tif_name).replace('.tif', '.shp'))





