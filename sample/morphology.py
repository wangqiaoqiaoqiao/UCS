import os
import time

import cv2
import numpy as np
from PIL import Image
import tifffile as tif

if __name__ == "__main__":
    data_dir = r'Y:\Training_Data\Test\Image_final\sample'
    negative_dir = r'Y:\Training_Data\Test\Image_final\sample_pred\Negative_pred_50'
    output_dir = r'Y:\Training_Data\Test\Image_final\sample_pred\Negative_pred_50_morphology'
    colors = [[0, 0, 0], [255, 255, 255], [38, 38, 205], [34, 139, 34], [255, 191, 0]]  # 黑 白 黄 绿
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            if (name != "Thumbs.db"):
                if "Thumbs.db" not in name:
                    start = time.time()
                    negative_label = np.array(Image.open(os.path.join(negative_dir, name[:-4] + '.png')))
                    k = np.ones((7, 7), np.uint8)
                    opening = cv2.morphologyEx(negative_label, cv2.MORPH_OPEN, k)  # 开运算
                    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, k) # 闭运算

                    img = np.array(Image.open(os.path.join(data_dir, name[:-4] + '.tif')).convert('RGB'))
                    cv2.imwrite(os.path.join(output_dir, name[:-4] + '.png'), closing)
                    closing[closing == 1] = 255
                    cv2.imwrite(os.path.join(output_dir, name[:-4] + '_c.png'), closing)
                    combine = cv2.addWeighted(img, 0.5,
                                              np.array(Image.fromarray(closing.astype('uint8')).convert('RGB')), 0.5,
                                              0)
                    tif.imwrite(os.path.join(output_dir, name[:-4] + '.tif'), combine)
                    end = time.time()
                    print(end - start)