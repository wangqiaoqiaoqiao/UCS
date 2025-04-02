import os
import shutil

import cv2
import numpy as np
from PIL import Image as Image

if __name__ == "__main__":
    data_dir = r'Y:\Training_Data\Test\Image_final\sample'
    negative_dir = r'Y:\Training_Data\Test\Image_final\sample_pred\Negative_pred_50_morphology'
    build_dir = r'Y:\Training_Data\Test\Image_final\sample_pred\build_pred'
    ESB_dir = r'Y:\Training_Data\Test\Image_final\sample_pred\ESB_pred_SAM_50'
    GPC_dir = r'Y:\Training_Data\Test\Image_final\sample_pred\GPC_pred'
    output_dir = r'Y:\Training_Data\Test\Image_final\sample_pred\Label'
    colors = [[0,0,0],[255,255,255],[38,38,205],[34,139,34],[255,191,0]] # 黑 白 黄 绿
    os.makedirs(output_dir,exist_ok=True)
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            if (name != "Thumbs.db"):
                if "Thumbs.db" not in name:
                    negative_label = np.array(Image.open(os.path.join(negative_dir, name[:-4] + '.png')))
                    build_label = np.array(Image.open(os.path.join(build_dir, name[:-4] + '.png')))
                    final_label = np.ones(build_label.shape, dtype='uint8') * 255
                    final_label[negative_label == 1] = 0
                    # if os.path.exists(os.path.join(ESB_dir,name[:-4]+'.tif')):
                    ESB_label = np.array(Image.open(os.path.join(ESB_dir,name[:-4]+'.tif')))
                    final_label[ESB_label == 1] = 1 # ESB 1
                    if os.path.exists(os.path.join(GPC_dir,name[:-4]+'.png')):
                        GPC_label = np.array(Image.open(os.path.join(GPC_dir,name[:-4]+'.png')))
                        final_label[GPC_label == 1] = 2 # GPC 2
                    final_label[build_label == 1] = 3  # building 3
                    cv2.imwrite(os.path.join(output_dir,name[:-4]+'.png'),final_label)

                    color_SRIWS = np.zeros([final_label.shape[0],final_label.shape[1],3],dtype='uint8')
                    color_SRIWS[final_label == 0, :] = colors[0]
                    color_SRIWS[final_label == 255, :] = colors[1]
                    color_SRIWS[final_label == 1, :] = colors[2]
                    color_SRIWS[final_label == 2, :] = colors[3]
                    color_SRIWS[final_label == 3, :] = colors[4]
                    cv2.imwrite(os.path.join(output_dir, name[:-4] + '_c.png'), color_SRIWS)




