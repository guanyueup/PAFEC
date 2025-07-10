import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import concurrent.futures
import os
import argparse

def process_frame(frame_pair):
    frame1, frame2 = frame_pair
    # 转灰度
    frame1_resized = cv2.resize(frame1, (1920, 1080))
    frame2_resized = cv2.resize(frame2, (1920, 1080))
    
    gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)
    
    # 计算 SSIM
    ssim_score = ssim(gray1, gray2, data_range=gray2.max() - gray2.min())

    # 计算 PSNR
    psnr_score = cv2.PSNR(frame1_resized, frame2_resized)

    return ssim_score, psnr_score

def calculate_metrics_multithread(video1_path, video2_path, max_workers=8):
    # 打开两个视频
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    frame_pairs = []
    frame_idx = 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # if frame1.shape != frame2.shape:
        #     print(f"Warning: Frame size mismatch at frame {frame_idx}, skipping...")
        #     continue

        frame_pairs.append((frame1, frame2))
        frame_idx += 1

    cap1.release()
    cap2.release()

    #print(f"共读取 {len(frame_pairs)} 帧，开始多线程计算...")

    ssim_list = []
    psnr_list = []

    # 多线程处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_frame, frame_pairs))

    for ssim_score, psnr_score in results:
        ssim_list.append(ssim_score)
        psnr_list.append(psnr_score)

    #print(f"计算完成！")
    # print(f"平均 SSIM: {np.mean(ssim_list):.4f}")
    # print(f"平均 PSNR: {np.mean(psnr_list):.2f} dB")

    #return ssim_list, psnr_list
    return np.mean(ssim_list), np.mean(psnr_list), ssim_list, psnr_list


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--lr',default=1)
    parser.add_argument('--pos',default=1)
    parser.add_argument('--index',default=1)
    parser.add_argument('--len',default=1)
    return parser.parse_args()
# 示例
if __name__ == "__main__":
    
    args = arg_parse()
    video1 = "/home/zzp/FEC_Test/conference_test_burst_test/raw_video/raw_test_video.mp4"  # 原始视频
    #video_path = ["/home/zzp/FEC_Test/conference_test_rr=ssim<0.95","/home/zzp/FEC_Test/conference_test_before","/home/zzp/FEC_Test/conference_test_rr=0","/home/zzp/FEC_Test/conference_test_rr=10lr"]        # 经处理/受损视频
    video_path = []#['/home/zzp/FEC_Test/360P_CAFEC','/home/zzp/FEC_Test/360P_FEC','/home/zzp/FEC_Test/480P_CAFEC','/home/zzp/FEC_Test/480P_FEC','/home/zzp/FEC_Test/540P_CAFEC','/home/zzp/FEC_Test/540P_FEC','/home/zzp/FEC_Test/720P_CAFEC','/home/zzp/FEC_Test/720P_FEC']
    video_path.append(args.path)
    pos = int(args.pos)
    frame_in = int(args.index)
    loss_len = int(args.len)
    loss_rate = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    red_rate = [0.035, 0.07, 0.105, 0.14, 0.175, 0.21, 0.245, 0.28, 0.315, 0.35]
    red_rate_all =  [0.05 , 0.1,   0.15, 0.2,   0.25, 0.3,   0.35, 0.4,   0.45, 0.5 ]
    if int(args.lr) == 1:
        red_rate_list = red_rate_all
    else:
        red_rate_list = red_rate
    #print(video_path,red_rate_list)
    for path in video_path:
        with open(f"{path}/more_loss/ssim_psnr_{frame_in}_{pos}_{loss_len}.csv", 'w') as f:
            f.write('\n')
            for lr,rr in zip([10],[10]):
                file_name = f'lossrate_{lr}_redundancy_{rr}.mp4'
                video2 = os.path.join(path,file_name)
                if not (os.path.exists(video1) and os.path.exists(video2)):
                    print("请确保两个视频文件存在于当前目录！")
                else:
                    ssim_vals, psnr_vals, ssim_list, psnr_list = calculate_metrics_multithread(video1, video2, max_workers=18)
                    f.write(f'{ssim_vals},{psnr_vals}\n')
                    for ss,ps in zip(ssim_list, psnr_list):
                        f.write(f'{ss},{ps}\n')



