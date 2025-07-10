import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def get_list(file_name,l):
    status_list = pd.read_csv(file_name,header=None)[1].tolist()
    len_list = []
    for i in range(len(status_list)-l):
        count = 0
        for j in range(i,i+l):
            if status_list[j] == 1:
                count += 1
        len_list.append(count)
    return len_list
    
#for l in range(20,100,10):
for loss in np.linspace(0.01,0.1,10):
    loss = round(loss,3)
    file_name1 = f'{loss}_loss_record.csv'
    file_name2 = f'{loss}_loss_record_o.csv'
    l1 = get_list(file_name1,32)
    l2 = get_list(file_name2,32)
    l1 = [i for i in l1 if i>0]
    l2 = [i for i in l2 if i>0]
    
    count1 = Counter(l1)
    count2 = Counter(l2)
    
      # 计算总数（用于计算百分比）
    total1 = len(l1)
    total2 = len(l2)
    # 获取所有出现过的数字（合并两组数据的所有唯一值）
    all_numbers = sorted(set(l1 + l2))

    # 为每个数字准备计数数据（如果某个数字在某组中没出现，计数为0）
    counts1 = [count1.get(num, 0) for num in all_numbers]
    counts2 = [count2.get(num, 0) for num in all_numbers]
    
     # 计算百分比
    percentages1 = [(count/total1)*100 if total1 > 0 else 0 for count in counts1]
    percentages2 = [(count/total2)*100 if total2 > 0 else 0 for count in counts2]
    
    result = []
    result2 = []
    current_sum = 0
    current_sum2 = 0
    for num,num2 in zip(percentages1,percentages2):
        current_sum += num
        current_sum2 += num2
        current_sum = round(current_sum,2)
        current_sum2 = round(current_sum2,2)
        result.append(current_sum)
        result2.append(current_sum2)
    print(result)
    print(result2)
    
    sum1 = [counts1[count] if count+1 <=5 else 0 for count in range(len(counts1))]
    sum2 = [counts2[count] if count+1 <=5 else 0 for count in range(len(counts2))]
    p1 = (np.sum(sum1) / total1)*100
    p2 = (np.sum(sum2) / total2)*100
    print(p1,p2)
    
    # 创建柱状图
    x = np.arange(len(all_numbers))
    width = 0.35
    print(x)
    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, counts1, width, label='GE', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, counts2, width, label='mine', color='lightcoral', alpha=0.8)


        # 百分比显示在柱子内部
    def add_percentage_labels_inside(bars, percentages):
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{percentage:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height/2),
                        ha='center', va='center',
                        fontsize=9,
                        color='black',
                        fontweight='bold')

    add_percentage_labels_inside(bars1, percentages1)
    add_percentage_labels_inside(bars2, percentages2)
    # 设置标签和标题
    ax.set_xlabel('number')
    ax.set_ylabel('count')
    ax.set_title('count analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(all_numbers)
    ax.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

