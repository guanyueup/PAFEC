import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress



def analyze_data(raw_data):
    loss_rate = raw_data[0].to_list() #+ raw_sFEC[0].to_list()
    ssim = raw_data[1].to_list() #+ raw_sFEC[1].to_list()
    psnr = raw_data[2].to_list() #+ raw_sFEC[2].to_list()

    ssim_sum_dic ={}
    psnr_sum_dic ={}
    for i,l in enumerate(loss_rate):
        ssim_sum_dic.setdefault(l,0)
        ssim_sum_dic[l] += ssim[i]
        psnr_sum_dic.setdefault(l,0)
        psnr_sum_dic[l] += psnr[i]
        
    for k in ssim_sum_dic.keys():
        ssim_sum_dic[k] /= len(ssim) / 10
        psnr_sum_dic[k] /= len(psnr) / 10
    return ssim_sum_dic.values(), psnr_sum_dic.values()

x = np.linspace(0.01,0.1,10)

ssim_ll_CAFEC = []
ssim_ll_FEC = []

# for muti in np.linspace(0.3,0.6,4):
#     muti = round(muti,1)
raw_CAFEC = pd.read_csv(f'ssim_psnr_CAFEC.csv',header=None)#_{muti}
raw_sFEC = pd.read_csv(f'ssim_psnr_FEC.csv',header=None)
ssim_cafec,psnr_cafec = analyze_data(raw_CAFEC)
ssim_sfec,psnr_sfec = analyze_data(raw_sFEC)
    # ssim_ll_CAFEC.append(ssim_cafec)
    # ssim_ll_FEC.append(ssim_sfec)


    # remainder = 0
    # for s1,s2 in zip(ssim_cafec,ssim_sfec):
    #     remainder += s1 -s2
    # print(remainder / 5)

    # ssim_l1_s = pd.read_csv('ssim_psnr_1_6_s.csv',header=None)[0].tolist()[1:101]
    # ssim_l1_m = pd.read_csv('ssim_psnr_1_6.csv',header=None)[0].tolist()[1:101]
    # x = np.arange(len(ssim_l1_m))

        # ``'b'``          blue
        #     ``'g'``          green
        #     ``'r'``          red
        #     ``'c'``          cyan
        #     ``'m'``          magenta
        #     ``'y'``          yellow
        #     ``'k'``          black
        #     ``'w'``          white

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制第一条折线
plt.plot(x, ssim_cafec, label='CAFEC', color='blue', linestyle='-', marker='o')

# 绘制第二条折线
plt.plot(x, ssim_sfec, label='FEC', color='red', linestyle='--', marker='s')

# # 绘制第三条折线
# plt.plot(x, ssim_ll_CAFEC[2], label='0.5_CAFEC', color='green', linestyle='-.', marker='^')

# # 绘制第三条折线
# plt.plot(x, ssim_ll_CAFEC[3], label='0.6_CAFEC', color='gray', linestyle='--', marker='*')

# # 绘制第一条折线
# plt.plot(x, ssim_ll_FEC[0], label='0.3_FEC', color='cyan', linestyle='-', marker='o')

# # 绘制第二条折线
# plt.plot(x, ssim_ll_FEC[1], label='0.4_FEC', color='black', linestyle='--', marker='s')

# # 绘制第三条折线
# plt.plot(x, ssim_ll_FEC[2], label='0.5_FEC', color='magenta', linestyle='-.', marker='^')

# # 绘制第三条折线
# plt.plot(x, ssim_ll_FEC[3], label='0.6_FEC', color='black', linestyle='--', marker='*')

# 添加标题和标签
plt.title('loss——ssim ', fontsize=16)
plt.xlabel('lossrate', fontsize=12)
plt.ylabel('ssim', fontsize=12)

# 添加图例
plt.legend()

# 显示网格
plt.grid(True, linestyle='--', alpha=0.5)

# 显示图形
plt.show()