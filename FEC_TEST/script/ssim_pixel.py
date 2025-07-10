import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor
import os



def caculate_ssim(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化变量
    prev_frame_gray = None
    frame_count = 0
    file_name = video_path.split('/')[-1].split('.')[0]
    with open(f'/home/zzp/FEC_Test/video_dataset_ssim_pixel/{file_name}.csv','w') as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 将当前帧转换为灰度图像
            current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 计算当前帧与前一帧的 SSIM
            if prev_frame_gray is not None:

                score, _ = ssim(prev_frame_gray, current_frame_gray, full=True)
                diff = cv2.absdiff(prev_frame_gray, current_frame_gray)
                pixeldiff_sum = np.sum(diff)
                squared_diff = np.square(diff)
                mse = np.mean(squared_diff)
                f.write(f'{score},{pixeldiff_sum},{mse}\n')
                #print(f"Frame {frame_count}: SSIM = {score:.4f}")

            # 更新前一帧
            prev_frame_gray = current_frame_gray
            frame_count += 1

        # 释放视频捕获对象
    cap.release()


if __name__ == '__main__':
    name_list = ['th','th-bb','th-m','th-ob']
    root_path = '/home/zzp/VCD/download/LOCAL_PATH/mp4'
    path_list = []
    for n in name_list:
        file_path = os.path.join(root_path,n)
        file_list = os.listdir(file_path)
        for f in file_list:
            path_list.append(os.path.join(file_path,f))
    #caculate_ssim(path_list[0])
    with ProcessPoolExecutor(max_workers=18) as executor:
        executor.map(caculate_ssim, path_list)
