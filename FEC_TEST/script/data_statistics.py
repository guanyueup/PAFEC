import pandas as pd
import os

root_path = '/home/zzp/FEC_Test/video_dataset_ssim_pixel'
ssim_list = []
mse_list = []

file_list = os.listdir(root_path)

for file in file_list:
    path = os.path.join(root_path,file)
    raw_data = pd.read_csv(path,header=None)
    ssim_list += raw_data[0].tolist()
    mse_list += raw_data[2].tolist()

count = 0
for s in ssim_list:
    if float(s) >= 0.95:
        count += 1
print(count,len(ssim_list),count/len(ssim_list))