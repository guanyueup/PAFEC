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

loss_list = np.linspace(0.01,0.1,10)
loss_list = [round(lr,2) for lr in loss_list]

ssim_ll_CAFEC = []
ssim_ll_FEC = []

# for muti in np.linspace(0.3,0.6,4):
#     muti = round(muti,1)
raw_CAFEC = pd.read_csv(f'ssim_psnr_CAFEC.csv',header=None)#_{muti}
raw_sFEC = pd.read_csv(f'ssim_psnr_FEC.csv',header=None)

#raw_CAFEC_3 = pd.read_csv(f'/Users/zhengzhaopeng/Desktop/FEC/muti_lr_1.0_muti_rr_1.5_muti_pace_1.0_count_pace_12_p_ratio_0.1_bottom_0.6_alpha_0.5_1080P_ca/ssim_psnr_FEC.csv',header=None)#_{muti}
#raw_CAFEC_5 = pd.read_csv(f'/Users/zhengzhaopeng/Desktop/FEC/muti_lr_1.0_muti_rr_1.5_muti_pace_1.0_count_pace_12_p_ratio_0.1_bottom_0.6_alpha_0.5_1080P_flex/ssim_psnr_FEC.csv',header=None)#_{muti}
#raw_CAFEC_7 = pd.read_csv(f'/Users/zhengzhaopeng/Desktop/FEC/muti_lr_1.0_muti_rr_1.5_muti_pace_1.0_count_pace_12_p_ratio_0.1_bottom_0.6_alpha_0.5_1080P_tooth/ssim_psnr_FEC.csv',header=None)#_{muti}

raw_CAFEC_3 = pd.read_csv(f'/Users/zhengzhaopeng/Documents/PAFEC/script/result_data/low_ssim_psnr_720P_PAFEC.csv',header=None)#_{muti}
raw_CAFEC_5 = pd.read_csv(f'/Users/zhengzhaopeng/Documents/PAFEC/script/result_data/low_ssim_psnr_720P_FlexFEC.csv',header=None)#_{muti}
raw_CAFEC_7 = pd.read_csv(f'/Users/zhengzhaopeng/Documents/PAFEC/script/result_data/low_ssim_psnr_720P_Tooth.csv',header=None)#_{muti}


ssim_cafec,psnr_cafec = analyze_data(raw_CAFEC)
ssim_sfec,psnr_sfec = analyze_data(raw_sFEC)

ssim_cafec3,psnr_cafec3 = analyze_data(raw_CAFEC_3)
ssim_cafec5,psnr_cafec5 = analyze_data(raw_CAFEC_5)
ssim_cafec7,psnr_cafec7 = analyze_data(raw_CAFEC_7)


# 设置柱状图的宽度和位置
x = np.arange(len(loss_list))  # x 轴的位置
bar_width = 0.25  # 每个柱的宽度

#添加水平线
# for y in np.linspace(5,40,8):
#     plt.axhline(y=y, color='gray', linestyle='--', linewidth=1.5,zorder=0)

print(np.max([(n1-n2) for n1,n2 in zip(psnr_cafec3,psnr_cafec5)]))
print(np.mean([(n1-n2) for n1,n2 in zip(psnr_cafec3,psnr_cafec5)]))

print(np.max([(n1-n2) for n1,n2 in zip(psnr_cafec3,psnr_cafec7)]))
print(np.mean([(n1-n2) for n1,n2 in zip(psnr_cafec3,psnr_cafec7)]))
# # 绘制柱状图
# plt.bar(x + bar_width, ssim_cafec3, width=bar_width, label='CAFEC', color='green')
# plt.bar(x - bar_width, ssim_cafec5, width=bar_width, label='FlexFEC', color='blue')
# plt.bar(x, ssim_cafec7, width=bar_width, label='Tooth', color='orange')



# # 添加 x 轴标签和标题
# plt.xlabel('loss rate')
# plt.ylabel('psnr')
# #plt.title('Bar Chart Example')



# # 添加图例
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# # 显示图表
# plt.tight_layout()  # 自动调整布局
# plt.show()
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
plt.plot(x, ssim_cafec3, label='CAFEC', color='green', linestyle='-', marker='o')

# 绘制第二条折线
plt.plot(x, ssim_cafec5, label='FlexFEC', color='blue', linestyle='--', marker='s')

# # 绘制第三条折线
plt.plot(x, ssim_cafec7, label='Tooth', color='orange', linestyle='-.', marker='^')

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
# 设置 x 轴的刻度和标签
plt.xticks(x, loss_list)
#plt.title('loss——ssim ', fontsize=16)
plt.xlabel('loss rate', fontsize=12)
plt.ylabel('psnr', fontsize=12)

# 添加图例
plt.legend()

# 显示网格
plt.grid(True, linestyle='--', alpha=0.5)

# 显示图形
plt.show()