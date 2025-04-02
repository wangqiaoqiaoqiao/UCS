import geopandas as gpd
import os
import pandas
from osgeo import ogr
from shapely import wkt
import shapefile as shp
def simplify_geometry(geometry, tolerance):
    """
    Simplifies the input geometry using the Douglas-Peucker algorithm.
    """
    ogr_geometry = ogr.CreateGeometryFromWkt(geometry.wkt)
    simplified_ogr_geometry = ogr_geometry.SimplifyPreserveTopology(tolerance)
    return wkt.loads(simplified_ogr_geometry.ExportToWkt())
def merge_ESB(shp_dir, shp_ESB):
    rootdir = shp_dir
    savedir_ESB = shp_ESB + '.shp'
    if not os.path.exists((rootdir)):
        return
    if os.path.exists(savedir_ESB):
        return
    # 创建一个空的GeoDataFrame来存储所有的矢量数据
    gdfs_ESB = []
    # 用循环读取所有的shp文件
    listfile = [os.path.join(rootdir, f) for f in os.listdir(rootdir) if f.endswith('.shp')]
    print('共有：', len(listfile))
    flag = 0
    print('开始', savedir_ESB)
    for filepath in listfile:
        # 读取矢量数据
        gdf = gpd.read_file(filepath)
        flag = flag + len(gdf[gdf['value'] == 1])
        gdfs_ESB.append(gdf[gdf['value'] == 1])
        # gdf['geometry'] = gdf['geometry'].apply(lambda geom: simplify_geometry(geom, tolerance))
    merged_gdf_ESB = gpd.GeoDataFrame(pandas.concat(gdfs_ESB, ignore_index=True), crs=gpd.read_file(listfile[0]).crs)
    merged_gdf_ESB.to_file(savedir_ESB)
def merge_GPC(shp_dir, shp_ESB):
    rootdir = shp_dir
    savedir_ESB = shp_ESB + '.shp'
    if not os.path.exists((rootdir)):
        return
    if os.path.exists(savedir_ESB):
        return
    # 创建一个空的GeoDataFrame来存储所有的矢量数据
    gdfs_ESB = []
    # 用循环读取所有的shp文件
    listfile = [os.path.join(rootdir, f) for f in os.listdir(rootdir) if f.endswith('.shp')]
    print('共有：', len(listfile))
    flag = 0
    print('开始', savedir_ESB)
    for filepath in listfile:
        # 读取矢量数据
        gdf = gpd.read_file(filepath)
        flag = flag + len(gdf[gdf['value'] == 2])
        gdfs_ESB.append(gdf[gdf['value'] == 2])
    merged_gdf_ESB = gpd.GeoDataFrame(pandas.concat(gdfs_ESB, ignore_index=True), crs=gpd.read_file(listfile[0]).crs)
    merged_gdf_ESB.to_file(savedir_ESB)
if __name__ == '__main__':
    print('开始：')
    rootpath = rf'Y:\Training_Data\Test\Final_result\Label\shp'
    shp_ESB = rf'Y:\Training_Data\Test\Final_result\UCS\Label\ESB'
    shp_GPC = rf'Y:\Training_Data\Test\Final_result\UCS\Label\GPC'
    merge_ESB(rootpath, shp_ESB)
    merge_GPC(rootpath, shp_GPC)
    print('结束:')