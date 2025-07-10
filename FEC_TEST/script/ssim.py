import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import subprocess


raw_frame_path = '/home/zzp/FEC_Test/conference_frame'
new_frame_path = '/home/zzp/FEC_Test/new_frame'


file_count = len([1 for f in os.listdir(raw_frame_path) if os.path.isfile(os.path.join(raw_frame_path, f))])

with open('pixel_conference.csv','w') as f: 
    f.write('0\n')
    for i in range(1,file_count,1):
        # 读取图片
        path1 = f'{raw_frame_path}/frame_{i:04}.png'
        path2 = f'{raw_frame_path}/frame_{i+1:04}.png'
        image1 = cv2.imread(path1)
        image2 = cv2.imread(path2)

        # 转换为灰度图像
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # 计算 SSIM 值
        #ssim_value, _ = ssim(image1_gray, image2_gray, full=True)
        
        diff = cv2.absdiff(image1_gray, image2_gray)
        pixeldiff_sum = np.sum(diff)
        #squared_diff = np.square(diff)
        #mse = np.mean(squared_diff)
        
        f.write(str(pixeldiff_sum)+'\n') 
        
        # if(ssim_value >0.85):
        #     subprocess.run(f'cp {path1} {new_frame_path}/frame_{i+1:04}.png',shell=True)
        # else:
        #     print(1)
        #     subprocess.run(f'cp {path2} {new_frame_path}/frame_{i:04}.png',shell=True)
            

        #输出 SSIM 值
        #print(f"SSIM between the two images: {i},{i+1}", ssim_value)
