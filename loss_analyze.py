import pandas as pd
import numpy as np
import ast
from collections import Counter
import os
import matplotlib.pyplot as plt

flex_path = './muti_lr_1.0_muti_rr_1.5_muti_pace_1.0_count_pace_12_p_ratio_0.1_bottom_0.6_alpha_0.5_1080P_flex/'
tooth_path = './muti_lr_1.0_muti_rr_1.5_muti_pace_1.0_count_pace_12_p_ratio_0.1_bottom_0.6_alpha_0.5_1080P_tooth/'
ca_path = './muti_lr_1.0_muti_rr_1.5_muti_pace_1.0_count_pace_12_p_ratio_0.1_bottom_0.6_alpha_0.5_1080P_ca/'
path_list = [flex_path,tooth_path,ca_path]
def get_data(file):
    list_1 = []
    list_2 = []
    list_3 = []
    with open(file,'r') as f:
        l = f.readline()
        while l:
            l_list = l.split(',')
            list_1.append(l_list[1])
            if len(l_list) > 3:
                loss_list = ast.literal_eval(','.join(l_list[2:]))
                list_2.append(len(loss_list))
                if 0 in loss_list:
                    list_3.append(1)
                else:
                    list_3.append(0)
            else:
                list_2.append(len(ast.literal_eval(l_list[2])))
                list_3.append(0)
            l= f.readline()
    return list_1, list_2, list_3



def grouping(data_list: list):
    first_half = []
    second_half = []
    view_len = 324
    pack_num = [data_list[i:i+view_len] for i in range(0,len(data_list),view_len)]
    for sublist in pack_num:
        for i in range(1,len(sublist),25):
            first_half.append(sublist[i:i+12])
            second_half.append(sublist[i+12:i+24])
    return first_half, second_half
 

loss_list = np.linspace(0.01,0.1,10)   
loss_list = [round(lr,2) for lr in loss_list]     

flex_first = []
tooth_first = []
ca_first = []
flex_second = []
tooth_second = []
ca_second = []

for lr in loss_list:
    for path in path_list:
        lr = round(lr,2)
        file_path = os.path.join(path,f'not_repair_num_{lr}.csv')
        l1,l2,l3 = get_data(file_path)
        first_half, second_half = grouping(l3)
        if path ==ca_path:
            ca_first.append(np.mean([np.mean(e) for e in first_half]))
            ca_second.append(np.mean([np.mean(e) for e in second_half]))
        elif path == flex_path:
            flex_first.append(np.mean([np.mean(e) for e in first_half]))
            flex_second.append(np.mean([np.mean(e) for e in second_half]))
        else:
            tooth_first.append(np.mean([np.mean(e) for e in first_half]))
            tooth_second.append(np.mean([np.mean(e) for e in second_half]))
            

np.min([(n1-n2)/n1*100 for n1,n2 in zip(flex_first,ca_first)])

# 设置柱状图的宽度和位置
x = np.arange(len(loss_list))  # x 轴的位置
bar_width = 0.25  # 每个柱的宽度

#添加水平线
for y in np.linspace(0.0025,0.0225,9):
    plt.axhline(y=y, color='gray', linestyle='--', linewidth=1.5,zorder=0)

# 绘制柱状图
plt.bar(x - bar_width, flex_first, width=bar_width, label='FlexFEC', color='blue')
plt.bar(x, tooth_first, width=bar_width, label='Tooth', color='orange')
plt.bar(x + bar_width, ca_first, width=bar_width, label='CAFEC', color='green')


# 添加 x 轴标签和标题
plt.xlabel('loss rate')
plt.ylabel('avg loss number')
#plt.title('Bar Chart Example')

# 设置 x 轴的刻度和标签
plt.xticks(x, loss_list)

# 添加图例
plt.legend()

# 显示图表
plt.tight_layout()  # 自动调整布局
plt.show()