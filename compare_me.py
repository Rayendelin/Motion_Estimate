"""
比较两种运动估计方法的优劣 两个评估指标 valid区域的占比 valid区域的PSNR 二者都是越高越好
"""
import os
from data_io import read_exr, write_exr
import numpy as np


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = -10 * np.log10(mse)
    return psnr


def main(raw_path, me_path):
    # 获取元数据
    buffer_names = sorted(os.listdir(me_path))
    index_start = int(buffer_names[0].split('.')[1])
    index_end = int(buffer_names[-1].split('.')[1])
    scene_name = me_path.split('/')[2]

    valid_percent_average = 0.0
    valid_psnr_average = 0.0
    total_psnr_average = 0.0
    compare_count = 0
    for index in range(index_start, index_end, 2):
        compare_count += 1

        warp_color_mask = read_exr(os.path.join(me_path, f"{scene_name}WarpColor.{str(index).zfill(4)}.exr"), channel=4)
        warp_color = warp_color_mask[:, :, :3]
        mask = warp_color_mask[:, :, 3:4]
        color_gt = read_exr(os.path.join(raw_path, f"{scene_name}PreTonemapHDRColor.{str(index).zfill(4)}.exr"), channel=3)
        
        invalid_area = np.sum(mask == 0)
        valid_percent = 1 - invalid_area / (warp_color.shape[0] * warp_color.shape[1])
        valid_percent_average += valid_percent

        warp_color = (warp_color ** (1/2.2)).clip(0, 1)
        color_gt = (color_gt ** (1/2.2)).clip(0, 1)
        total_psnr = calculate_psnr(warp_color, color_gt)
        total_psnr_average += total_psnr

        color_gt = color_gt * mask
        valid_psnr = calculate_psnr(warp_color, color_gt)
        valid_psnr_average += valid_psnr

    valid_percent_average /= compare_count
    valid_psnr_average /= compare_count
    total_psnr_average /= compare_count
    print("valid_percent_average: " + str(valid_percent_average))
    print("valid_psnr_average: " + str(valid_psnr_average))
    print("total_psnr_average: " + str(total_psnr_average))
    
    return valid_percent_average, valid_psnr_average, total_psnr_average


if __name__ == "__main__":
    raw_path = "E:/workspace/zwtdataset/Bunker/train1-60fps-combine"
    fgsr_me_path = "./fgsr_me/Bunker/train1-60fps-combine"
    gffe_me_path = "./gffe_me/Bunker/train1-60fps-combine"
    fuse_me_path = "./fuse_me/Bunker/train1-60fps-combine"

    main(raw_path, fgsr_me_path)
    main(raw_path, gffe_me_path)
    main(raw_path, fuse_me_path)