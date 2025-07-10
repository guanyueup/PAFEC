import pandas as pd
import ast
import numpy as np

file1 = 'not_repair_num_muti_lr_1.0_muti_rr_2.0_muti_pace_0.1_count_pace_1_p_ratio_0.6_bottom_0.5_CAFEC.csv'
file2 = 'not_repair_num_muti_lr_1.0_muti_rr_2.0_muti_pace_0.1_count_pace_1_p_ratio_0.6_bottom_0.5_FEC.csv'
def get_data(file):
    list_1 = []
    list_2 = []
    with open(file,'r') as f:
        l = f.readline()
        while l:
            l_list = l.split(',')
            list_1.append(l_list[1])
            if len(l_list) > 3:
                list_2.append(len(ast.literal_eval(','.join(l_list[2:]))))
                
            else:
                list_2.append(len(ast.literal_eval(l_list[2])))
            l= f.readline()
    return list_1, list_2
l1,l2 = get_data(file1)
l3,l4 = get_data(file2)

view_len = 324

for pos in range(10):
    pack_num = [l1[i:i+view_len] for i in range(pos*324,len(l1),3240)]
    loss_list = [l2[i:i+view_len] for i in range(pos*324,len(l2),3240)]
    loss_list1 = [l4[i:i+view_len] for i in range(pos*324,len(l4),3240)]

    import matplotlib.pyplot as plt
    y1 = [0]*view_len
    y2 = [0]*view_len
    for ll1,ll2 in zip(loss_list,loss_list1):
        y1 = np.add(y1, ll1)
        y2 = np.add(y2, ll2)
    y1 = y1.astype(float)
    y2 = y2.astype(float)
    y1 /= len(loss_list)
    y2 /= len(loss_list1)
    
    # 数据
    categories = pack_num[1]# 横坐标类别
    
    x = np.arange(len(categories))
    width = 0.35  # 柱子宽度

    fig, ax = plt.subplots(figsize=(10, 6))

    # 创建柱状图
    bars1 = ax.bar(x - width/2, y1, width, label='CAFEC', color='skyblue')
    bars2 = ax.bar(x + width/2, y2, width, label='FEC', color='lightcoral')

    # 添加标签和标题
    ax.set_xlabel('index')
    ax.set_ylabel('loss_avg')
    ax.set_title('inde-loss_avg')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # 在柱子上显示数值
    # def autolabel(bars):
    #     for bar in bars:
    #         height = bar.get_height()
    #         ax.annotate(f'{height}',
    #                     xy=(bar.get_x() + bar.get_width() / 2, height),
    #                     xytext=(0, 3),  # 3 points vertical offset
    #                     textcoords="offset points",
    #                     ha='center', va='bottom')

    # autolabel(bars1)
    # autolabel(bars2)

    plt.tight_layout()
    plt.show()

    # # 绘制柱状图
    # plt.bar(categories, values, color='skyblue')

    # # 添加标题和标签
    # plt.title('inde_loss', fontsize=16)
    # plt.xlabel('index', fontsize=12)
    # plt.ylabel('loss', fontsize=12)

    # # 显示图表
    # plt.show()