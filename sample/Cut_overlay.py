import math
import os
from osgeo import gdal
if __name__ == '__main__':
    data_dir = r'Y:\Training_Data\Bu_Image\Xinjiang'
    for root, dirs, files in os.walk(data_dir):
        for name in files:
                        data = gdal.Open(os.path.join(root, name))
                        im_width = data.RasterXSize  # 行数
                        im_height = data.RasterYSize  # 列数
                        # ##定义切图的大小（矩形框）
                        block_xsize = 1024  # 行
                        block_ysize = 1024  # 列
                        overlay = int(1024 / 16)
                        num_width = math.ceil(im_width / (block_xsize - overlay))
                        num_height = math.ceil(im_height / (block_ysize - overlay))
                        if data.RasterCount == 3:
                            in_band1 = data.GetRasterBand(1)
                            in_band2 = data.GetRasterBand(2)
                            in_band3 = data.GetRasterBand(3)
                        elif data.RasterCount == 1:
                            in_band1 = data.GetRasterBand(1)
                        geo_information = data.GetGeoTransform()
                        top_left_x = geo_information[0]  # 左上角x坐标
                        top_left_y = geo_information[3]  # 左上角y坐标
                        w_e_pixel_resolution = geo_information[1]  # 东西方向像素分辨率
                        n_s_pixel_resolution = geo_information[5]  # 南北方向像素分辨率
                        image = data.ReadAsArray()
                        # ##############################裁切信息设置###################################
                        # ##定义切图的起始点像素位置
                        offset_x = 0
                        offset_y = 0
                        count = 0

                        for i in range(0, im_width, block_xsize - overlay):
                            offset_x1 = offset_x + i
                            if offset_x1 + block_xsize > im_width:
                                offset_x1 = im_width - block_xsize
                            for j in range(0, im_height, block_ysize - overlay):
                                count = count + 1
                                offset_y1 = offset_y + j
                                if offset_y1 + block_ysize > im_height:
                                    offset_y1 = im_height - block_ysize
                                if data.RasterCount == 3:
                                    out_band1 = in_band1.ReadAsArray(offset_x1, offset_y1, block_xsize, block_ysize)
                                    out_band2 = in_band2.ReadAsArray(offset_x1, offset_y1, block_xsize, block_ysize)
                                    out_band3 = in_band3.ReadAsArray(offset_x1, offset_y1, block_xsize, block_ysize)
                                elif data.RasterCount == 1:
                                    out_band1 = in_band1.ReadAsArray(offset_x1, offset_y1, block_xsize, block_ysize)
                                gtif_driver = gdal.GetDriverByName("GTiff")
                                filename = os.path.join(r'Y:\Training_Data\Image_CutOverlay', name[:-4] + '_' + str(count) + '.tif')  # 文件名称
                                out_ds = gtif_driver.Create(filename, block_xsize, block_ysize, int(data.RasterCount),
                                                            in_band1.DataType)  # 数据格式遵循原始图像

                                top_left_x1 = top_left_x + offset_x1 * w_e_pixel_resolution
                                top_left_y1 = top_left_y + offset_y1 * n_s_pixel_resolution

                                dst_transform = (
                                    top_left_x1, geo_information[1], geo_information[2], top_left_y1, geo_information[4],
                                    geo_information[5])
                                out_ds.SetGeoTransform(dst_transform)

                                # 设置SRS属性（投影信息）
                                out_ds.SetProjection(data.GetProjection())

                                # 写入目标文件（如果波段数有更改，这儿也需要修改）
                                if data.RasterCount == 3:
                                    out_ds.GetRasterBand(1).WriteArray(out_band1)
                                    out_ds.GetRasterBand(2).WriteArray(out_band2)
                                    out_ds.GetRasterBand(3).WriteArray(out_band3)
                                elif data.RasterCount == 1:
                                    out_ds.GetRasterBand(1).WriteArray(out_band1)
                                # 将缓存写入磁盘，直接保存到了程序所在的文件夹
                                out_ds.FlushCache()
                                print(f"FlushCache succeed{count}")
                                del out_ds