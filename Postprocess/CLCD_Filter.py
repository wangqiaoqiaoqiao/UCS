import os
import arcpy
# 'Jiangsu''Shandong', 'Liaolin',
imagedirs = ['Neimenggu']

# 34个省份的字典
provinces = {
    'Beijing': 'beijing',
    'Tianjin': 'tianjin',
    'Hebei': 'hebei',
    'Shanxi_T': 'shanxi',
    'Neimenggu': 'neimeng',
    'Liaolin': 'niaoning',
    'Jilin': 'jining',
    'Heilongjiang': 'heilongjiang',
    'Shanghai': 'shanghai',
    'Jiangsu': 'jiangsu',
    'Zhejiang': 'zhejiang',
    'Anhui': 'anhui',
    'Fujian': 'fujian',
    'Jiangxi': 'jiangxi',
    'Shandong': 'shandong',
    'Henan': 'henan',
    'Hubei': 'hubei',
    'Hunan': 'hunan',
    'Guangdong': 'guangzhou',
    'Guangxi': 'guangxi',
    'Hainan': 'hainan',
    'Chongqing': 'chongqing',
    'Sichuan': 'sichuang',
    'Guizhou': 'guizhou',
    'Yunnan': 'yunnan',
    'Xizang': 'xizang',
    'Shanxi_X': 'shaanxi',
    'Gansu': 'ganshu',
    'Qinghai': 'qinghai',
    'Ningxia': 'ningxia',
    'Xinjiang': 'xinjiang',
    'Taiwan': 'taiwan',
    'Xianggang': 'hongkong',
    'Aomen': 'macao'
}

# 确保临时文件夹存在
temp_folder = r'Y:\Training_Data\Bu_Image_Pred\Shp\Shp_CLCD\temp'
os.makedirs(temp_folder, exist_ok=True)
for name in imagedirs:
    shps = [f[:-4] for f in
            os.listdir(os.path.join("X:\Image_Bu", name, "City_filter")) if f.endswith('.shp')]
    for shp_city in shps[8:]:
        target_shapefile_ESB = os.path.join(r'Y:\Training_Data\Bu_Image_Pred\Shp\Shp_dissolved',
                                            name, shp_city + '_' + 'ESB' + '_dissolved.shp')
        target_shapefile_GPC = os.path.join(r'Y:\Training_Data\Bu_Image_Pred\Shp\Shp_dissolved',
                                            name, shp_city + '_' + 'GPC' + '_dissolved.shp')
        output_folder = os.path.join(r'Y:\Training_Data\Bu_Image_Pred\Shp\Shp_CLCD', name)
        os.makedirs(output_folder, exist_ok=True)
        output_shapefile_ESB = os.path.join(output_folder, shp_city + '_' + 'ESB' + '.shp')
        output_shapefile_GPC = os.path.join(output_folder, shp_city + '_' + 'GPC' + '.shp')

        if arcpy.Exists(output_shapefile_ESB) and arcpy.Exists(output_shapefile_GPC):
            continue
        if not arcpy.Exists(target_shapefile_ESB) and not arcpy.Exists(target_shapefile_GPC):
            continue

        try:
            print('开始创建', name)

            # 初始化当前的shapefile
            current_shapefile_ESB = target_shapefile_ESB
            current_shapefile_GPC = target_shapefile_GPC
            values = [1, 2, 3, 4, 6, 7, 9]

            for value in values:
                clcd_path = os.path.join(r'Y:\Training_Data\CLCD\2021_2022_CLCD1',
                                         'CLCD_v01_albert_' + provinces.get(name, 'Other') + '_' + str(value) + '.shp')
                output_clcd_projected = os.path.join(r'Y:\Training_Data\CLCD\2021_2022_CLCD_pro',
                                                     'CLCD_v01_albert_' + provinces.get(name, 'Other') + '_' + str(
                                                         value) + '.shp')
                if os.path.exists(clcd_path):
                    if not os.path.exists(output_clcd_projected):
                        # 获取 CLCD.shp 的投影
                        clcd_spatial_ref = arcpy.Describe(clcd_path).spatialReference

                        # 定义 CLCD.shp 的投影为其自身的投影
                        arcpy.management.DefineProjection(clcd_path, clcd_spatial_ref)

                        # 获取 ESB.shp 的投影
                        esb_spatial_ref = arcpy.Describe(current_shapefile_ESB).spatialReference

                        # 将 CLCD.shp 投影到 ESB.shp 的投影
                        arcpy.management.Project(clcd_path, output_clcd_projected, esb_spatial_ref)
                        print(f"Projected {clcd_path} to {output_clcd_projected}")

                    # 创建临时图层
                    polygon_layer = "polygon_layer"
                    target_layer_ESB = "target_layer_ESB"
                    target_layer_GPC = "target_layer_GPC"

                    # 检查并删除已存在的临时图层
                    if arcpy.Exists(polygon_layer):
                        arcpy.Delete_management(polygon_layer)
                    if arcpy.Exists(target_layer_ESB):
                        arcpy.Delete_management(target_layer_ESB)
                    if arcpy.Exists(target_layer_GPC):
                        arcpy.Delete_management(target_layer_GPC)

                    arcpy.MakeFeatureLayer_management(output_clcd_projected, polygon_layer)
                    arcpy.MakeFeatureLayer_management(current_shapefile_ESB, target_layer_ESB)
                    arcpy.MakeFeatureLayer_management(current_shapefile_GPC, target_layer_GPC)

                    # 选择与面要素相交的目标要素
                    arcpy.SelectLayerByLocation_management(in_layer=target_layer_ESB, overlap_type="WITHIN",
                                                           select_features=polygon_layer)
                    arcpy.SelectLayerByLocation_management(in_layer=target_layer_GPC, overlap_type="WITHIN",
                                                           select_features=polygon_layer)

                    # 反转选择
                    arcpy.SelectLayerByLocation_management(in_layer=target_layer_ESB, selection_type="SWITCH_SELECTION")
                    arcpy.SelectLayerByLocation_management(in_layer=target_layer_GPC, selection_type="SWITCH_SELECTION")

                    # 将反转选择结果保存到临时 Shapefile
                    temp_esb_folder = os.path.join(temp_folder, name, 'ESB')
                    os.makedirs(temp_esb_folder, exist_ok=True)
                    esb_no_intersection_ESB = os.path.join(temp_esb_folder, f'{shp_city}_esb_no_intersection_{value}.shp')
                    arcpy.CopyFeatures_management(target_layer_ESB, esb_no_intersection_ESB)
                    print(f"Saved non-intersecting features to {esb_no_intersection_ESB}")

                    current_shapefile_ESB = esb_no_intersection_ESB

                    temp_gpc_folder = os.path.join(temp_folder, name, 'GPC')
                    os.makedirs(temp_gpc_folder, exist_ok=True)
                    esb_no_intersection_GPC = os.path.join(temp_gpc_folder, f'{shp_city}_esb_no_intersection_{value}.shp')
                    arcpy.CopyFeatures_management(target_layer_GPC, esb_no_intersection_GPC)
                    print(f"Saved non-intersecting features to {esb_no_intersection_GPC}")

                    current_shapefile_GPC = esb_no_intersection_GPC

            clcd5_path = os.path.join(r'Y:\Training_Data\CLCD\2021_2022_CLCD1',
                                      'CLCD_v01_albert_' + provinces.get(name, 'Other') + '_5.shp')
            output_clcd5_projected = os.path.join(r'Y:\Training_Data\CLCD\2021_2022_CLCD_pro',
                                                  'CLCD_v01_albert_' + provinces.get(name, 'Other') + '_5.shp')
            if os.path.exists(clcd5_path):
                if not os.path.exists(output_clcd5_projected):
                    # 获取 CLCD.shp 的投影
                    clcd_spatial_ref = arcpy.Describe(clcd5_path).spatialReference

                    # 定义 CLCD.shp 的投影为其自身的投影
                    arcpy.management.DefineProjection(clcd5_path, clcd_spatial_ref)

                    # 获取 ESB.shp 的投影
                    esb_spatial_ref = arcpy.Describe(current_shapefile_ESB).spatialReference

                    # 将 CLCD.shp 投影到 ESB.shp 的投影
                    arcpy.management.Project(clcd5_path, output_clcd5_projected, esb_spatial_ref)
                    print(f"Projected {clcd5_path} to {output_clcd5_projected}")

                containing_features_layer_ESB = "containing_features_layer_ESB"
                containing_features_layer_GPC = "containing_features_layer_GPC"
                containing_polygon_layer = "containing_polygon_layer"

                # 删除已存在的图层
                if arcpy.Exists(containing_features_layer_ESB):
                    arcpy.Delete_management(containing_features_layer_ESB)
                if arcpy.Exists(containing_features_layer_GPC):
                    arcpy.Delete_management(containing_features_layer_GPC)
                if arcpy.Exists(containing_polygon_layer):
                    arcpy.Delete_management(containing_polygon_layer)

                arcpy.MakeFeatureLayer_management(current_shapefile_ESB, containing_features_layer_ESB)
                arcpy.MakeFeatureLayer_management(current_shapefile_GPC, containing_features_layer_GPC)
                arcpy.MakeFeatureLayer_management(output_clcd5_projected, containing_polygon_layer)

                arcpy.SelectLayerByLocation_management(in_layer=containing_features_layer_ESB, overlap_type='WITHIN',
                                                       select_features=containing_polygon_layer)
                arcpy.SelectLayerByLocation_management(in_layer=containing_features_layer_GPC, overlap_type='INTERSECT',
                                                       select_features=containing_polygon_layer)

                # 反转选择
                arcpy.SelectLayerByLocation_management(in_layer=containing_features_layer_ESB,
                                                       selection_type="SWITCH_SELECTION")
                arcpy.SelectLayerByLocation_management(in_layer=containing_features_layer_GPC,
                                                       selection_type="SWITCH_SELECTION")

                current_shapefile_ESB = containing_features_layer_ESB
                current_shapefile_GPC = containing_features_layer_GPC

            # 保存结果
            arcpy.management.CopyFeatures(current_shapefile_ESB, output_shapefile_ESB)
            arcpy.management.CopyFeatures(current_shapefile_GPC, output_shapefile_GPC)

            print("处理完成，结果已保存到：", output_shapefile_ESB, output_shapefile_GPC)

        except arcpy.ExecuteError:
            print(arcpy.GetMessages(2))
        except Exception as e:
            print(f"发生未处理的异常：{e}")