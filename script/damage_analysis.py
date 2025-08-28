import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

from ploter import IEEEPlotter
import os

plotter = IEEEPlotter(base_size=8, column='single', color_mode='auto')



#图1
def test2():
    file_on = '/Users/zhengzhaopeng/Desktop/FEC/sigle_loss_econ/ssim_psnr_1_3_1.csv'
    file_off = '/Users/zhengzhaopeng/Desktop/FEC/sigle_loss_ecoff/ssim_psnr_1_3_1.csv'
    raw_data_on = pd.read_csv(file_on,header=None)
    raw_data_off = pd.read_csv(file_off,header=None)
    y1 = raw_data_on[0].tolist()[1:]
    y2 = raw_data_off[0].tolist()[1:]
    x = np.linspace(1,len(y1),len(y1))
    
    
    # 2) 折线图（自动颜色/线型/标记）
    fig, ax = plotter.line(
        x=x,
        y_list=[ y1, y2],
        labels=['EC ON', 'EC OFF'],
        xlabel='Frame Index', ylabel='Frame SSIM',
        title=''
    )
    plotter.save(fig, 'GoP_loss_affect.pdf')

#图2
def test1():
    frame_list_1 = []
    frame_list_10 = []

    for i in range(1,10):
        package_list_1 = []
        package_list_10 = []
        for j in range(1,10):
            file_path1 = f'../more_loss_ecoff/ssim_psnr_{i}_{j}_10.csv'
            file_path2 = f'../more_loss_econ/ssim_psnr_{i}_{j}_10.csv'
            raw_data_1 = pd.read_csv(file_path1,header=None)
            raw_data_10 = pd.read_csv(file_path2,header=None)
            package_list_1.append(raw_data_1[0][0])
            package_list_10.append(raw_data_10[0][0])
        frame_list_1.append(package_list_1)
        frame_list_10.append(package_list_10)
        

    x = np.linspace(1,9,9)
    y1 = frame_list_1[3]
    y2 = frame_list_10[3]
    # y = [package_list_1[0] for package_list_1 in frame_list_1]
    # y1 = [package_list_10[1] for package_list_10 in frame_list_1]
    # y2 = [package_list_10[1] for package_list_10 in frame_list_10]

    
    # 2) 折线图（自动颜色/线型/标记）
    fig, ax = plotter.line(
        x=x,
        y_list=[ y1, y2],
        labels=['EC OFF', 'EC ON'],
        xlabel='Package Index', ylabel='Average SSIM',
        title=''
    )
    plotter.save(fig, 'EC_affect_high.pdf')

    
#图3
def test4():
    dir_path = '/Users/zhengzhaopeng/Desktop/FEC/video_dataset_ssim_pixel'