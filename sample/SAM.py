import shutil
import time

from segment_anything import SamPredictor, sam_model_registry
import tifffile as tif
from PIL import Image
import numpy as np
import torch
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

if __name__ == "__main__":
    data_dir = r'Y:\Training_Data\Test\Image_final\sample'
    mask_dir = r'Y:\Training_Data\Test\Image_final\sample_pred\ESB_pred_50'
    sam_checkpoint = "D:\wwr\Second\sample\checkpoint\sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    output_path = r'Y:\Training_Data\Test\Image_final\sample_pred\ESB_pred_SAM_50'
    os.makedirs(output_path, exist_ok=True)
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            if (name != "Thumbs.db"):
                start = time.time()
                if os.path.exists(os.path.join(output_path, name[:-4] + '_c.tif')):
                    continue
                else:
                    image = Image.open(os.path.join(data_dir, name[:-4] + '.tif')).convert('RGB')
                    mask = Image.open(os.path.join(mask_dir, name[:-4] + '.png'))
                    mask_dense = Image.open(os.path.join(mask_dir, name[:-4] + '_logits.tif'))
                    fg_point = np.argwhere(np.array(mask) == 1)
                    if len(fg_point) == 0:
                        # continue
                        mask = np.array(Image.open(os.path.join(mask_dir, name[:-4] + '.png')))
                        tif.imwrite(os.path.join(output_path, name[:-4] + '.tif'), mask)
                        mask[mask == 1] = 255
                        tif.imwrite(os.path.join(output_path, name[:-4] + '_c.tif'), mask)
                    else:
                        fg_point = fg_point[np.random.randint(len(fg_point), size=20)]
                        fg_label = np.ones(len(fg_point), dtype='uint8')
                        fg_point = fg_point[:, [-1, 0]]
                        bg_point = np.argwhere(np.array(mask) == 0)
                        bg_point = bg_point[np.random.randint(len(bg_point), size=10)]
                        bg_label = np.zeros(len(bg_point), dtype='uint8')
                        bg_point = bg_point[:, [-1, 0]]
                        point_sample = np.append(fg_point, bg_point, axis=0)
                        point_label = np.append(fg_label, bg_label, axis=0)

                        mask_dense = mask_dense.resize((int(mask_dense.height * 0.25), int(mask_dense.width * 0.25)),
                                                       resample=Image.NEAREST)
                        predictor.set_image(np.array(image))  # 设置要分割的图像
                        mask_dense = np.array(mask_dense)
                        masks, _, _ = predictor.predict(
                            point_coords=point_sample,
                            point_labels=point_label,
                            mask_input=mask_dense[None, :, :],
                            multimask_output=False,
                            return_logits=True
                        )
                        tif.imwrite(os.path.join(output_path, name[:-4] + '_dense.tif'), masks.squeeze())
                        fg_point = np.argwhere(np.array(mask) == 1)
                        fg_point = fg_point[np.random.randint(len(fg_point), size=15)]
                        fg_label = np.ones(len(fg_point), dtype='uint8')
                        fg_point = fg_point[:, [-1, 0]]
                        bg_point = np.argwhere(np.array(mask) == 0)
                        bg_point = bg_point[np.random.randint(len(bg_point), size=20)]
                        bg_label = np.zeros(len(bg_point), dtype='uint8')
                        bg_point = bg_point[:, [-1, 0]]
                        point_sample = np.append(fg_point, bg_point, axis=0)
                        point_label = np.append(fg_label, bg_label, axis=0)
                        masks_fianl, _, _ = predictor.predict(
                            point_coords=point_sample,
                            point_labels=point_label,
                            mask_input=torch.nn.functional.interpolate(torch.from_numpy(masks).unsqueeze(0),
                                                                       256, mode="bilinear", align_corners=False).squeeze(
                                0).numpy(),
                            multimask_output=False,
                        )
                        masks_fianl = masks_fianl[0].astype('uint8')
                        tif.imwrite(os.path.join(output_path, name[:-4] + '.tif'), masks_fianl)
                        masks_fianl[masks_fianl == 1] = 255
                        combine = cv2.addWeighted(np.array(image), 0.5, np.array(Image.fromarray(masks_fianl).convert('RGB')),
                                                  0.5, 0)
                        tif.imwrite(os.path.join(output_path, name[:-4] + '_show.tif'), combine)
                        tif.imwrite(os.path.join(output_path, name[:-4] + '_c.tif'), masks_fianl)
                end = time.time()
                print(end-start)

