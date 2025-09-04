import numpy as np
import matplotlib.pyplot as plt
from script.ploter import IEEEPlotter

raw_len = 6222
ca_mine_size = [7944,8389,8608,8724,9051,9819,10015,10254,10414,10828]
flex_mine_size = [8119,8414,8704,9041,9271,9933,10283,10559,10847,11157]
tooth_mine_size = [7979,8380,8529,8804,9168,9590,9961,10322,10689,10837]

ca_ge_size = [7708,8324,8933,8051,9047,9517,9143,9686,9316,9411]
flex_ge_size = [8119,8414,8704,9041,9271,9933,10283,10559,10847,11157]
tooth_ge_size = [7979,8380,8529,8804,9168,9590,9961,10322,10689,10837]

# ca_mine_size = [4559,5017,5296,5428,5466,5548,5588,5626,5666,5708]
# flex_mine_size = [4260,4404,4543,4748,4797,5164,5357,5474,5620,5801]
# tooth_mine_size = [4759,5298,5475,5917,6094,6772,6949,7126,7595,7804]

size_list = [ca_mine_size,ca_ge_size,flex_mine_size,tooth_mine_size]

loss_list = np.linspace(0.01,0.1,10)
loss_list = [round(lr,2) for lr in loss_list]

ca_mine = []
ca_ge = []
flex = []
tooth = []

for lr in loss_list:
    for sl in size_list:
        if sl == ca_mine_size:
            ca_mine = [round((s-raw_len)/raw_len *100,2) for s in sl]
        elif sl == ca_ge_size:
            ca_ge = [round((s-raw_len)/raw_len *100,2) for s in sl]
        elif sl == flex_mine_size:
            flex = [round((s-raw_len)/raw_len *100,2) for s in sl]
        elif sl == tooth_mine_size:
            tooth = [round((s-raw_len)/raw_len *100,2) for s in sl]
            
print(np.max([(n2-n1)/n2 * 100 for n1,n2 in zip(ca_mine,flex)]))

# 设置柱状图的宽度和位置
x = np.arange(len(loss_list))  # x 轴的位置
bar_width = 0.25  # 每个柱的宽度

#添加水平线
for y in np.linspace(10,80,9):
    plt.axhline(y=y, color='gray', linestyle='--', linewidth=1.5,zorder=0)

# 绘制柱状图
plt.bar(x - bar_width, flex, width=bar_width, label='FlexFEC', color='blue')
plt.bar(x, tooth, width=bar_width, label='Tooth', color='orange')
plt.bar(x + bar_width, ca_mine, width=bar_width, label='CAFEC', color='green')


# 添加 x 轴标签和标题
plt.xlabel('loss rate')
plt.ylabel('redundancy rate')
#plt.title('Bar Chart Example')

# 设置 x 轴的刻度和标签
plt.xticks(x, loss_list)

# 添加图例
plt.legend()

# 显示图表
plt.tight_layout()  # 自动调整布局
plt.show()