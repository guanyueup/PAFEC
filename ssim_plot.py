import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

def plot():
    x = np.linspace(0.005,0.05,10)


    CAFEC_ssim = pd.read_csv('ssim_psnr_f_CAFEC.csv',header=None)[1].tolist()
    CAFEC_psnr = pd.read_csv('ssim_psnr_f_CAFEC.csv',header=None)[2].tolist()
    FEC_ssim = pd.read_csv('ssim_psnr_f_FEC.csv',header=None)[1].tolist()
    FEC__psnr = pd.read_csv('ssim_psnr_f_FEC.csv',header=None)[2].tolist()

    # 创建图形
    plt.figure(figsize=(8, 6))

    # 绘制第一条折线
    plt.plot(x, FEC_psnr, label='FEC_psnr', color='blue', linestyle='-', marker='o')

    # 绘制第二条折线
    plt.plot(x, CAFEC__psnr, label='CAFEC_psnr', color='red', linestyle='--', marker='s')

    # # 绘制第三条折线
    # plt.plot(x, CAFEC_540_psnr, label='CAFEC_540P', color='green', linestyle='-.', marker='^')

    # # 绘制第三条折线
    # plt.plot(x, FEC_540_psnr, label='FEC_540P', color='gray', linestyle='--', marker='*')

    # 添加标题和标签
    plt.title('psnr', fontsize=16)
    plt.xlabel('loss rate', fontsize=12)
    plt.ylabel('psnr', fontsize=12)

    # 添加图例
    plt.legend()

    # 显示网格
    plt.grid(True, linestyle='--', alpha=0.5)

    # 显示图形
    plt.show()