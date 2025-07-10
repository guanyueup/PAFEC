import pandas as pd
import os

siti_path = '/home/zzp/FEC_Test/siti'
frame_info = pd.read_csv('frame_info.csv')
file_list = list(frame_info['file_name'])
I_num = list(frame_info['I_num'])
I_size = list(frame_info['I_size'])
P_num = list(frame_info['P_num'])
P_size = list(frame_info['P_size'])
B_num = list(frame_info['B_num'])
B_size = list(frame_info['B_size'])


def get_avg_si_ti(data):
    # Calculate the average SI and TI
    avg_si = data['si'].mean()
    avg_ti = data['ti'].mean()
    return avg_si, avg_ti

with open('frame_siti.csv', 'w') as f:
    f.write('file_name, I_num, I_size, P_num, P_size, B_num, B_size, avg_si, avg_ti\n')
    for i,file in enumerate(file_list):
        file_path = os.path.join(siti_path,file.split('.')[0]+'.csv')
        avsi,avti = get_avg_si_ti(pd.read_csv(file_path))
        f.write(f"{file},{I_num[i]},{I_size[i]},{P_num[i]},{P_size[i]},{B_num[i]},{B_size[i]},{avsi:.3f},{avti:.3f}\n")
        