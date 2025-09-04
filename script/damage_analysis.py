import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

from ploter import IEEEPlotter
import os
import ast

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
    file_list = os.listdir(dir_path)

    count = 0
    avg_ssim_list = []


    for file in file_list:
        file_path = os.path.join(dir_path,file)
        raw_data = pd.read_csv(file_path,header=None)
        ssim_list = raw_data[0].tolist()
        avg_ssim_list.append(np.nanmean(ssim_list))

    # 4) CDF（经验分布）
    fig, ax = plotter.cdf([avg_ssim_list], labels=[''],
                          xlabel='Avg SSIM', title='')
    plotter.save(fig, 'ssim_CDF.pdf')
#图4
def test6():
    more_ssim_list = []
    more_psnr_list = []
    sigle_ssim_list = []
    sigle_psnr_list = []
    for pos in range(1,10):
        idx_ssim_list = []
        idx_psnr_list = []
        idx_ssim_list2 = []
        idx_psnr_list2 = []
        for idx in range(1,10):
            sigle_path = f'./fig4data/sigle_loss/ssim_psnr_{pos}_{idx}_1.csv'
            more_path = f'./fig4data/more_loss/ssim_psnr_{pos}_{idx}_10.csv'
            raw_data = pd.read_csv(more_path,header=None)
            raw_data2 = pd.read_csv(sigle_path,header=None)
            idx_ssim_list.append(raw_data[0][0])
            idx_psnr_list.append(raw_data[0][1])
            idx_ssim_list2.append(raw_data2[0][0])
            idx_psnr_list2.append(raw_data2[0][1])
        more_ssim_list.append(idx_ssim_list)
        more_psnr_list.append(idx_psnr_list)
        sigle_ssim_list.append(idx_ssim_list2)
        sigle_psnr_list.append(idx_psnr_list2)

    x = np.linspace(1,9,9)
    y1 = [ idx[1] for idx in more_ssim_list]
    y2 = [ idx[1] for idx in sigle_ssim_list]
    
     # 1) 柱状图（分组，自动颜色+纹理；含误差线与数值标签）
    cats = ['1','2','3','4','5','6','7','8','9']
    fig, ax = plotter.bar(
        categories=cats,
        series_list=[y1,y2],
        labels=['Loss 10', 'Loss 1'],
        yerr=None,
        show_values=False,
        ylabel='Avg SSIM', xlabel='Frame Index',
        title='',
        ylim=(0.95,0.98)
    )
    plotter.save(fig, 'Loss_om_affect.pdf')


    
#图8
def test5():
    x = pd.read_csv('./fig8data/mse_pixel_conference.csv',header=None)[0].to_list()[1:]
    y = pd.read_csv('./fig8data/ssim_conference.csv',header=None)[0].to_list()[1:]
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    # 3) 散点图（带线性拟合）
    fig, ax = plotter.scatter(x, y, label='Samples', fit='linear',
                              xlabel='Pixel mse', ylabel='SSIM', title='',fit_color='red')
    plotter.save(fig, 'conffee.pdf')

#图12 图13
def test7():
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
        return list(ssim_sum_dic.values()), list(psnr_sum_dic.values())
    loss_list = np.linspace(0.01,0.1,10)
    loss_list = [round(lr,2) for lr in loss_list]
    x = np.arange(len(loss_list))  # x 轴的位置
    raw_CAFEC_3 = pd.read_csv(f'./fig9data/ssim_psnr_CAFEC-0.3.csv',header=None)#_{muti}
    raw_CAFEC_5 = pd.read_csv(f'./fig9data/ssim_psnr_CAFEC-0.5.csv',header=None)#_{muti}
    raw_CAFEC_7 = pd.read_csv(f'./fig9data/ssim_psnr_CAFEC-0.7.csv',header=None)#_{muti}
    
    ssim_cafec3,psnr_cafec3 = analyze_data(raw_CAFEC_3)
    ssim_cafec5,psnr_cafec5 = analyze_data(raw_CAFEC_5)
    ssim_cafec7,psnr_cafec7 = analyze_data(raw_CAFEC_7)
    
    print(np.max([(n1-n2) for n1,n2 in zip(psnr_cafec3,psnr_cafec5)]))
    print(np.mean([(n1-n2) for n1,n2 in zip(psnr_cafec3,psnr_cafec5)]))

    print(np.max([(n1-n2) for n1,n2 in zip(psnr_cafec3,psnr_cafec7)]))
    print(np.mean([(n1-n2) for n1,n2 in zip(psnr_cafec3,psnr_cafec7)]))
    
    
    
    # 2) 折线图（自动颜色/线型/标记）
    fig, ax = plotter.line(
        x=x,
        y_list=[ ssim_cafec3, ssim_cafec5,ssim_cafec7],
        labels=['Beta=0.3', 'Beta=0.5', 'Beta=0.5'],
        xlabel='Loss Rate', ylabel='Average SSIM',
        title=''
    )
    plotter.save(fig, 'B_ssim.pdf')

#fig10
def test8():
    flex_path = './fig10data/not_repair_FlexFEC1/'
    tooth_path = './fig10data/not_repair_Tooth1/'
    ca_path = './fig10data/not_repair_PAFEC1/'
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
        view_len = 300
        pack_num = [data_list[i:i+view_len] for i in range(0,len(data_list),view_len)]
        for sublist in pack_num:
            for i in range(1,len(sublist),30):
                first_half.append(sublist[i:i+14])
                second_half.append(sublist[i+14:i+29])
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
                

    print(np.min([(n1-n2)/n1*100 for n1,n2 in zip(flex_second,ca_second)]))
    
    # 1) 柱状图（分组，自动颜色+纹理；含误差线与数值标签）
    cats = np.linspace(0.01,0.1,10)
    cats = [round(lr,2) for lr in cats ]
    fig, ax = plotter.bar(
        categories=cats,
        series_list=[flex_first,tooth_first,ca_first],
        labels=['FlexFEC', 'Tooth', 'SPFEC'],
        yerr=None,
        show_values=False,
        ylabel='Avg Loss Number', xlabel='Loss Rate',
        title='',
        #ylim=(0.95,0.98)
    )
    plotter.save(fig, 'first_loss_num.pdf')
#fig11
def test9():
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
                
    print(np.min([(n2-n1)/n2 * 100 for n1,n2 in zip(ca_mine,flex)]))

    
    # 1) 柱状图（分组，自动颜色+纹理；含误差线与数值标签）
    cats = np.linspace(0.01,0.1,10)
    cats = [round(lr,2) for lr in cats ]
    fig, ax = plotter.bar(
        categories=cats,
        series_list=[flex,tooth,ca_mine],
        labels=['FlexFEC', 'Tooth', 'SPFEC'],
        yerr=None,
        show_values=False,
        ylabel='Redundancy Rate', xlabel='Loss Rate',
        title='',
        ylim=(20,80)
    )
    plotter.save(fig, 'video_size.pdf')
#图9
def test10():
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
        return list(ssim_sum_dic.values()), list(psnr_sum_dic.values())
    loss_list = np.linspace(0.01,0.1,10)
    loss_list = [round(lr,2) for lr in loss_list]
    x = np.arange(len(loss_list))  # x 轴的位置
    raw_CAFEC_3 = pd.read_csv(f'./fig9data/ssim_psnr_CAFEC-0.3.csv',header=None)#_{muti}
    raw_CAFEC_5 = pd.read_csv(f'./fig9data/ssim_psnr_CAFEC-0.5.csv',header=None)#_{muti}
    raw_CAFEC_7 = pd.read_csv(f'./fig9data/ssim_psnr_CAFEC-0.7.csv',header=None)#_{muti}
    
    ssim_cafec3,psnr_cafec3 = analyze_data(raw_CAFEC_3)
    ssim_cafec5,psnr_cafec5 = analyze_data(raw_CAFEC_5)
    ssim_cafec7,psnr_cafec7 = analyze_data(raw_CAFEC_7)
    
    print(np.max([(n1-n2) for n1,n2 in zip(psnr_cafec3,psnr_cafec5)]))
    print(np.mean([(n1-n2) for n1,n2 in zip(psnr_cafec3,psnr_cafec5)]))

    print(np.max([(n1-n2) for n1,n2 in zip(psnr_cafec3,psnr_cafec7)]))
    print(np.mean([(n1-n2) for n1,n2 in zip(psnr_cafec3,psnr_cafec7)]))
    
    
    # 1) 柱状图（分组，自动颜色+纹理；含误差线与数值标签）
    cats = np.linspace(0.01,0.1,10)
    cats = [round(lr,2) for lr in cats ]
    fig, ax = plotter.bar(
        categories=cats,
        series_list=[ssim_cafec3,ssim_cafec5,ssim_cafec7],
        labels=['Beta=0.3', 'Beta=0.5', 'Beta=0.7'],
        yerr=None,
        show_values=False,
        ylabel='Average SSIM', xlabel='Loss Rate',
        title='',
        #ylim=(0.7,1.0),
    
    )
    ax.legend(loc='lower center',
          bbox_to_anchor=(0.5, 1.02),
          ncol=len(ax.get_legend_handles_labels()[1]),  # 全部放一行
          frameon=False)

    plotter.save(fig, 'B_ssim.pdf')
#图9
def test11():

    # 指数平滑： MLL_t = (1-β) * MLL_{t-1} + β * MLL_NEW_t
    def smooth_series(mll_new, beta, init=None):
        mll_new = np.asarray(mll_new, dtype=float)
        y = np.empty_like(mll_new)
        y[0] = mll_new[0] if init is None else float(init)
        for t in range(1, len(mll_new)):
            y[t] = (1.0 - beta) * y[t-1] + beta * mll_new[t]
        return y

    # 1) 构造原始输入（ground truth）
    T = 50
    x = np.arange(T)
    mll_new = np.ones(T)
    mll_new[20:25] = 6.0  # 20~24为高值，其他为1

    # 2) 计算不同 β 的平滑曲线
    betas = [0.1, 0.3, 0.5, 0.9]
    y_list = [smooth_series(mll_new, b) for b in betas]
    labels = [f"beta = {b}" for b in betas]

    # 3) 用你的接口画图
    fig, ax = plotter.line(
        x=x,
        y_list=y_list,
        labels=labels,
        xlabel='Time step',
        ylabel='Value',
        title=''
    )

    # 4) 叠加原始输入（虚线阶跃）
    ax.step(x, mll_new, where='post', linestyle='--', linewidth=2, label='Raw input (ground truth)')
    ax.legend(loc='upper right')

    # 5) 保存
    plotter.save(fig, 'mll_smoothing.pdf')


if __name__ == '__main__':
    test8()