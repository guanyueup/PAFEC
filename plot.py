import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


# 生成两个示例数据
ssim = pd.read_csv('/Users/zhengzhaopeng/Desktop/ssim_conference.csv',header=None)[0].tolist()
pixel_diff = pd.read_csv('/Users/zhengzhaopeng/Desktop/pixel_conference.csv',header=None)[0].tolist()
pixel_mse = pd.read_csv('/Users/zhengzhaopeng/Desktop/mse_pixel_conference.csv',header=None)[0].tolist()


slope, intercept, r_value, p_value, std_err = linregress(pixel_mse,ssim)

# 打印拟合结果
print(f"Slope (m): {slope}")
print(f"Intercept (b): {intercept}")
print(f"R-squared: {r_value**2}")

# 拟合直线
y_fit = [slope * xi + intercept for xi in pixel_mse]

# 绘制散布图
plt.scatter(pixel_diff, ssim, color='blue', alpha=0.7)
#plt.plot(pixel_mse, y_fit, color='red', label='Fitted Line')
plt.title("pixel diff sum correlation to SSIM")
plt.xlabel("pixel diff sum")
plt.ylabel("SSIM")
plt.grid(True)
plt.show()

#计算相关系数
correlation = np.corrcoef(pixel_diff,ssim)[0, 1]  # 计算皮尔逊相关系数
print(f"相关系数 (Correlation Coefficient): {correlation:.2f}")