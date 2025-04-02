import os
import shutil
import glob
import arcpy
def dissolve(srcshp, targetshp):
    arcpy.RepairGeometry_management(srcshp)
    arcpy.management.Dissolve(srcshp, targetshp, multi_part='SINGLE_PART')
def erase_overlap(input_shp, erase_shp, output_shp):
    """
    从 input_shp 中裁掉与 erase_shp 重叠的区域，并将结果保存到 output_shp。

    参数:
    input_shp - 需要裁剪的输入 Shapefile 路径
    erase_shp - 用作掩膜的 Shapefile 路径
    output_shp - 裁剪后保存的 Shapefile 路径
    """
    # 检查并删除已存在的输出文件
    if arcpy.Exists(output_shp):
        arcpy.Delete_management(output_shp)

    # 执行 Erase 操作
    arcpy.analysis.Erase(input_shp, erase_shp, output_shp)
    print(f"Output saved to {output_shp}")
def copy_shapefile(src_dir, dest_dir, shapefile_name):
    """
    复制 Shapefile 及其关联文件。
    """
    pattern = os.path.join(src_dir, shapefile_name + '.*')
    for src_file in glob.glob(pattern):
        dest_file = dest_dir + os.path.basename(src_file)[-4:]
        shutil.copy(src_file, dest_file)
def move_shapefile(src_dir, dest_dir, shapefile_name):
    """
    复制 Shapefile 及其关联文件。
    """
    pattern = os.path.join(src_dir, shapefile_name + '.*')
    for src_file in glob.glob(pattern):
        dest_file = dest_dir + os.path.basename(src_file)[-4:]
        shutil.move(src_file, dest_file)
def process_shapefiles(image, shapefiles, output_dirs):
    """
    处理 shapefiles，将重叠区域保留 value 最大的面要素。
    """
    # 设置临时文件路径
    temp_dir = r'Y:\Training_Data\Pred_China_Final\province_JSshp\temp\temp'
    os.makedirs(temp_dir,exist_ok=True)
    ESB = shapefiles[0]
    GPC = shapefiles[1]
    Build = shapefiles[2]
    projected_shapefile = output_dirs[2] + '.shp'
    # 获取 Shapefile 的空间参考
    spatial_ref = arcpy.Describe(Build).spatialReference
    target_sr = arcpy.Describe(ESB).spatialReference
    arcpy.management.DefineProjection(Build, spatial_ref)
    arcpy.management.Project(Build, projected_shapefile, target_sr)

    # 利用 GPC 擦除 ESB
    temp_shp1 = os.path.join(temp_dir, f"{image}_temp1.shp")
    erase_overlap(ESB, GPC, temp_shp1)

    # 利用 Build 擦除 ESB (先擦除 GPC 再擦除 Build)
    temp_shp2 = os.path.join(temp_dir, f"{image}_temp2.shp")
    erase_overlap(temp_shp1, projected_shapefile, temp_shp2)

    # 利用 Build 擦除 GPC
    temp_shp3 = os.path.join(temp_dir, f"{image}_temp3.shp")
    erase_overlap(GPC, projected_shapefile, temp_shp3)

    # 将结果复制到输出目录
    move_shapefile(temp_dir, output_dirs[0], f"{image}_temp2")
    move_shapefile(temp_dir, output_dirs[1], f"{image}_temp3")
def add_value_field_and_set_value(shapefile, value):
    # 添加 value 字段
    # 检查字段是否存在
    fields = arcpy.ListFields(shapefile, "value")
    if not fields:
        arcpy.management.AddField(shapefile, "value", "SHORT")

        # 更新 value 字段值
        with arcpy.da.UpdateCursor(shapefile, ["value"]) as cursor:
            for row in cursor:
                row[0] = value
                cursor.updateRow(row)
def merge_and_clip(shapefile_paths, output_folder, shapefile_path):
    arcpy.env.workspace = output_folder
    os.makedirs(output_folder, exist_ok=True)

    # 为第一个 Shapefile 添加 value 字段并设置为 1
    add_value_field_and_set_value(shapefile_paths[0], 1)

    # 为第二个 Shapefile 添加 value 字段并设置为 2
    add_value_field_and_set_value(shapefile_paths[1], 2)

    # 临时合并shapefile
    temp_merge = os.path.join(output_folder, f'{shapefile_path}_temp_merge.shp')
    if not arcpy.Exists(temp_merge):
        arcpy.management.Merge(shapefile_paths, temp_merge)
    if not arcpy.Exists(temp_merge):
        raise FileNotFoundError(f"临时合并文件 {temp_merge} 未创建")
    temp_merge_dis = os.path.join(output_folder, f'{shapefile_path}_temp_merge_dis.shp')
    if not arcpy.Exists(temp_merge_dis):
        dissolve(temp_merge, temp_merge_dis)
    return temp_merge_dis
def Polygon_hole(input_shapefile, output_shapefile):
    # 设置最小面积阈值（单位为输入数据的坐标单位，例如平方米）
    area_threshold = 100.0  # 小于此面积的孔洞将被填补
    percent_threshold = 10.0
    # 调用 Eliminate Polygon Part 工具
    arcpy.management.EliminatePolygonPart(
        in_features=input_shapefile,  # 输入要素
        out_feature_class=output_shapefile,  # 输出要素类
        condition="AREA_OR_PERCENT",  # 根据面积筛选
        part_area=area_threshold,
        part_area_percent=percent_threshold,  # 面积阈值
        part_option="CONTAINED_ONLY"  # 处理所有部分
    )

    print(f"孔洞已填补，结果保存到: {output_shapefile}")
def delete_area(output_shapefile):
    # 添加面积字段
    arcpy.AddField_management(output_shapefile, "area", "DOUBLE")

    # 计算面积并存储到 area 字段，单位为平方米
    arcpy.CalculateGeometryAttributes_management(output_shapefile, [["area", "AREA"]], "METERS", "SQUARE_METERS")
    print(f"{output_shapefile} 面积计算完成并存储在 'area' 字段中，单位为平方米。")
    # 创建图层
    arcpy.management.MakeFeatureLayer(output_shapefile, "layer")

    # 使用 SQL 表达式选择面积小于 10 平方米的面要素
    area_threshold = 10.0  # 阈值
    sql_expression = f'"area" < {area_threshold}'  # 替换 "Shape_Area" 为你的面积字段名
    arcpy.management.SelectLayerByAttribute("layer", "NEW_SELECTION", sql_expression)

    # 打印选中要素的数量
    selected_count = int(arcpy.management.GetCount("layer").getOutput(0))
    print(f"选中 {selected_count} 个面积小于 {area_threshold} 平方米的要素")

    # 删除选中的要素
    if selected_count > 0:
        arcpy.management.DeleteFeatures("layer")
        print("选中的要素已删除")
    else:
        print("没有符合条件的要素需要删除")

    # 清理图层
    arcpy.management.Delete("layer")
def add_source_fields_merge(shapefile, esb_shapefile, gpc_shapefile):
    # Add fields if they don't exist
    if not arcpy.ListFields(shapefile, "has_ESB"):
        arcpy.management.AddField(shapefile, "has_ESB", "SHORT")
    if not arcpy.ListFields(shapefile, "has_GPC"):
        arcpy.management.AddField(shapefile, "has_GPC", "SHORT")
    # Create feature layers and handle existing layers
    layers = ["shapefile_layer", "esb_layer", "gpc_layer"]
    for layer in layers:
        if arcpy.Exists(layer):
            arcpy.Delete_management(layer)
    # Create feature layers
    arcpy.management.MakeFeatureLayer(shapefile, "shapefile_layer")
    arcpy.management.MakeFeatureLayer(esb_shapefile, "esb_layer")
    arcpy.management.MakeFeatureLayer(gpc_shapefile, "gpc_layer")

    # Update cursor with detailed error checking
    with arcpy.da.UpdateCursor(shapefile, ["FID", "has_ESB", "has_GPC"]) as cursor:
        for row in cursor:
            fid = row[0]
            arcpy.management.SelectLayerByAttribute("shapefile_layer", "NEW_SELECTION", f"FID = {fid}")

            arcpy.management.SelectLayerByLocation("esb_layer", "WITHIN", "shapefile_layer")
            row[1] = 1 if int(arcpy.management.GetCount("esb_layer").getOutput(0)) > 0 else 0

            arcpy.management.SelectLayerByLocation("gpc_layer", "WITHIN", "shapefile_layer")
            row[2] = 1 if int(arcpy.management.GetCount("gpc_layer").getOutput(0)) > 0 else 0

            cursor.updateRow(row)

    print(f"已更新 {shapefile} 中的 has_ESB 和 has_GPC 字段")
def merge_close_polygons(temp_merge, output_folder, distance, shapefile_path):
    # 为每个多边形创建缓冲区
    temp_buffered = os.path.join(output_folder, f'{shapefile_path}_temp_buffered_{distance}m.shp')
    if not arcpy.Exists(temp_buffered):
        arcpy.analysis.Buffer(temp_merge, temp_buffered, f"{distance} Meters")
    if not arcpy.Exists(temp_buffered):
        raise FileNotFoundError(f"缓冲区文件 {temp_buffered} 未创建")

    # 合并缓冲区内相交的多边形
    temp_dissolved = os.path.join(output_folder, f'{shapefile_path}_temp_dissolved_{distance}m.shp')
    if not arcpy.Exists(temp_dissolved):
        arcpy.management.Dissolve(temp_buffered, temp_dissolved, multi_part='SINGLE_PART')
    if not arcpy.Exists(temp_dissolved):
        raise FileNotFoundError(f"溶解文件 {temp_dissolved} 未创建")

    # 检查 ID 字段是否存在，如果不存在则添加
    if not arcpy.ListFields(temp_dissolved, "ID"):
        arcpy.management.AddField(temp_dissolved, "ID", "LONG")

    # 使用 UpdateCursor 更新 ID 字段，赋值为 FID
    with arcpy.da.UpdateCursor(temp_dissolved, ["FID", "ID"]) as cursor:
        for row in cursor:
            row[1] = row[0]  # 将 FID 的值赋给 ID 字段
            cursor.updateRow(row)

    GD_shp = os.path.join(output_folder, f'{shapefile_path}_GD_{distance}m_final.shp')
    if not arcpy.Exists(GD_shp):
        # 创建临时文件用于存储空间连接结果
        temp_shp = os.path.join(os.path.dirname(GD_shp), rf"{shapefile_path}_{distance}m_temp_spatial_join.shp")

        # 执行空间连接，将 merge.shp 的面要素与 dissolve.shp 的多边形关联
        arcpy.analysis.SpatialJoin(
            target_features=temp_merge,
            join_features=temp_dissolved,
            out_feature_class=temp_shp,
            join_type="KEEP_ALL",  # 保留重叠部分
            match_option="WITHIN"  # 使用中心点在 dissolve.shp 内的要素
        )

        # 根据 dissolve.shp 的属性字段（如唯一 ID）对连接后的要素进行合并
        arcpy.management.Dissolve(
            in_features=temp_shp,
            out_feature_class=GD_shp,
            dissolve_field='ID_1'
        )

        # 删除临时文件
        arcpy.management.Delete(temp_shp)

        # calculate_area_by_source(temp_dissolved, projected_shapefiles[0], projected_shapefiles[1])
        # add_source_fields_merge(GD_shp, projected_shapefiles[0], projected_shapefiles[1])
        # 获取要素的数量
        count_result = arcpy.management.GetCount(GD_shp)
        count = int(count_result.getOutput(0))
        # 打印要素的数量
        print(f"Distance: {distance}")
        print(f"要素个数: {count}")

        print(f"合并完成，结果保存到: {GD_shp}")
    else:
        print(f"输出文件已存在: {GD_shp}")

if __name__ == '__main__':
    regions = ['Guangxi', 'Guizhou', 'Hainan',
               'Hebei', 'Heilongjiang', 'Henan']
    base_path = r'Y:\Training_Data\Pred_China_Final\province_city'
    output_folder = r'Y:\Training_Data\Pred_China_Final\province_JSshp'
    os.makedirs(output_folder, exist_ok=True)
    for region in regions:
        print(region, ':')
        shapefile_paths_list = [f[:-4] for f in os.listdir(os.path.join(r'Z:\Province', region, 'City')) if f.endswith('.shp') and f.startswith('feature')]
        for shapefile_path in shapefile_paths_list:
            if shapefile_path != 'feature_235':
                print(shapefile_path)
                if os.path.exists(rf'Y:\Training_Data\Pred_China_Final\province_city\temp_2020_2024\ESB_{shapefile_path}.shp'):
                    ESB_path = rf'Y:\Training_Data\Pred_China_Final\province_city\temp_2020_2024\ESB_{shapefile_path}.shp'
                elif os.path.exists(rf'Y:\Training_Data\Pred_China_Final\province_city\{region}\ESB_{shapefile_path}_NoMSK.shp'):
                    ESB_path = rf'Y:\Training_Data\Pred_China_Final\province_city\{region}\ESB_{shapefile_path}_NoMSK.shp'
                else:
                    ESB_path = rf'Y:\Training_Data\Pred_China_Final\province_city\{region}_error\ESB_{shapefile_path}_NoMSK.shp'
                print(ESB_path)
                if os.path.exists(rf'Y:\Training_Data\Pred_China_Final\province_city\{region}\GPC_{shapefile_path}.shp'):
                    GPC_path = rf'Y:\Training_Data\Pred_China_Final\province_city\{region}\GPC_{shapefile_path}.shp'
                else:
                    GPC_path = rf'Y:\Training_Data\Pred_China_Final\province_city\{region}_error\GPC_{shapefile_path}.shp'
                ESB_path_dis = ESB_path.replace('.shp', '_dis.shp')
                if not os.path.exists(ESB_path_dis):
                    print('开始ESB dissolve')
                    dissolve(ESB_path, ESB_path_dis)
                GPC_path_dis = GPC_path.replace('.shp', '_dis.shp')
                if not os.path.exists(GPC_path_dis):
                    print('开始GPC dissolve')
                    dissolve(GPC_path, GPC_path_dis)
                shapefiles = [
                    os.path.join(ESB_path_dis),
                    os.path.join(GPC_path_dis),
                    os.path.join(rf'Y:\Training_Data\Pred_China_Final\{region}\Build\{shapefile_path}.shp')
                ]
                output_dirs = [
                    rf'Y:\Training_Data\Pred_China_Final\province_JSshp\temp\ESB_{shapefile_path}',
                    rf'Y:\Training_Data\Pred_China_Final\province_JSshp\temp\GPC_{shapefile_path}',
                    rf'Y:\Training_Data\Pred_China_Final\province_JSshp\temp\Build_{shapefile_path}'
                ]
                if not os.path.exists(output_dirs[0] + '.shp') and not os.path.exists(output_dirs[1] + '.shp') and not \
                        os.path.exists(output_dirs[2] + '.shp'):
                    print('开始擦除')
                    process_shapefiles(shapefile_path, shapefiles, output_dirs)
                ESB = rf'Y:\Training_Data\Pred_China_Final\province_JSshp\temp\ESB_{shapefile_path}.shp'
                GPC = rf'Y:\Training_Data\Pred_China_Final\province_JSshp\temp\GPC_{shapefile_path}.shp'
                Build = rf'Y:\Training_Data\Pred_China_Final\province_JSshp\temp\Build_{shapefile_path}.shp'
                shapefile_paths = []
                shapefile_paths.append(ESB)
                shapefile_paths.append(GPC)
                region_output_folder = os.path.join(rf'Y:\Training_Data\Pred_China_Final\province_JSshp\{region}')
                temp_merge = merge_and_clip(shapefile_paths, region_output_folder, shapefile_path)
                output_path_nholes = rf'Y:\Training_Data\Pred_China_Final\province_JSshp\{region}\{shapefile_path}_JSGD_holes.shp'
                if not os.path.exists(output_path_nholes):
                    print('填补孔洞')
                    Polygon_hole(temp_merge, output_path_nholes)
                    delete_area(output_path_nholes)
                output_path_final = rf'Y:\Training_Data\morphological parameters\{region}'
                os.makedirs(output_path_final, exist_ok=True)
                distances = [10, 50, 100, 200]
                for distance in distances:
                    if not os.path.exists(os.path.join(output_path_final, f'{shapefile_path}_GD_{distance}m_final.shp')):
                        print('生成最终结果')
                        merge_close_polygons(output_path_nholes, output_path_final, distance, shapefile_path)

