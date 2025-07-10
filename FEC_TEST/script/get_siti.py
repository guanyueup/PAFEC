import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pandas as pd

video_path = '/home/zzp/youtubeucgdateset'
siti_path = '/home/zzp/FEC_Test/ssim'


file_list = os.listdir(video_path)

def siti(video_path, save_path):
    subprocess.run(f'siti -of csv -f {video_path}  > {save_path}',shell=True)
    
def get_ssim(video_path, save_path):  
    cap = cv2.VideoCapture(video_path)

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    # 存储每帧图像
    frames = []

    # 提取每一帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # 计算每一对帧之间的SSIM值
    ssim_matrix = np.zeros((total_frames, total_frames))
    with open(save_path, 'w') as f:
        for i in range(total_frames-1):
                # 将帧转换为灰度图像（SSIM计算通常在灰度图上进行）
                gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
                
                # 计算SSIM值
                ssim_value, _ = ssim(gray1, gray2,full=True)
                f.write(f"{ssim_value:.5f}\n")

video_path_list = []
siti_path_list = []
for file in file_list:
    if file.endswith('.mp4'):
        input_file = os.path.join(video_path, file)
        output_file = os.path.join(siti_path, file.split('.')[0]+'.csv')
        #get_ssim(input_file,output_file)
        video_path_list.append(input_file)
        siti_path_list.append(output_file)
with ProcessPoolExecutor(max_workers=18) as executor:
    executor.map(get_ssim, video_path_list, siti_path_list)