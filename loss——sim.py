import numpy as np
import random
import argparse
import os

import numpy as np

def simulate_burst_loss(num_packets, target_loss):
    """
    模拟具有目标丢包率的 Gilbert-Elliott 信道模型

    参数:
        num_packets (int): 要模拟的数据包数量
        target_loss (float): 目标丢包率（0.0 到 1.0）

    返回:
        losses (np.ndarray): 布尔数组，True 表示丢包，False 表示接收成功
    """
    # 固定好状态下和坏状态下的丢包率
    pg = 0.001  # Good 状态下几乎不丢
    pb = 0.8    # Bad 状态下严重丢

    # 反推出对应的状态转移概率 p 和 r
    # 设法使整体丢包率符合 target_loss
    best_diff = float('inf')
    best_p = best_r = None

    for p in np.linspace(0.001, 0.1, 50):
        for r in np.linspace(0.01, 0.5, 50):
            pi_g = r / (p + r)
            pi_b = p / (p + r)
            estimated_loss = pi_g * pg + pi_b * pb
            diff = abs(estimated_loss - target_loss)
            if diff < best_diff:
                best_diff = diff
                best_p, best_r = p, r

    # 模拟 Gilbert-Elliott 信道
    state = 'G'
    losses = []

    for _ in range(num_packets):
        if state == 'G':
            losses.append(np.random.rand() < pg)
            if np.random.rand() < best_p:
                state = 'B'
        else:  # state == 'B'
            losses.append(np.random.rand() < pb)
            if np.random.rand() < best_r:
                state = 'G'

    return np.array(losses)




def get_loss_num():
    # 数字列表
    numbers = np.arange(1,14)
    # 对应的概率，必须和数字列表长度相同，且总和为1
    probabilities = [0.372, 0.175, 0.14, 0.035, 0.035, 0.03, 0.03, 0.03, 0.015, 0.015, 0.015, 0.015, 0.093]
    # 使用 random.choices 选择一个数字，k=1 表示选择一个
    selected_number = random.choices(numbers, probabilities, k=1)[0]

    if selected_number==13:
        selected_number = random.randint(13,25)

    return selected_number

def loss_simulate(reslut_path, p_len,tag):
    loss_rate = np.linspace(0.01,0.1,10)
    
    for lr in loss_rate:
        lr =round(lr,3)
        loss_future = 0
        loss_all = 0
        current_packages = 0
        with open(f'{reslut_path}/network_trace/{tag}/{lr}_loss_record.csv','w') as f:
            for p in range(p_len):
                current_packages += 1
                if loss_future == 0:
                    if random.random() < lr:
                        loss_future = get_loss_num()
                        if loss_all <= current_packages * lr:
                            loss_future -= 1
                            loss_all += 1
                            f.write(str(current_packages)+','+str(1)+'\n')
                        else:
                            loss_future = 0
                            f.write(str(current_packages)+','+str(0)+'\n')
                    else:
                        f.write(str(current_packages)+','+str(0)+'\n')
                else:
                    loss_future -= 1
                    loss_all += 1
                    f.write(str(current_packages)+','+str(1)+'\n')
def loss_simulate1(reslut_path, p_len,tag):
    loss_rate = np.linspace(0.01,0.1,10)
    
    for lr in loss_rate:
        lr =round(lr,3)
        status = simulate_burst_loss(p_len,lr)
        with open(f'{reslut_path}/network_trace/{tag}/{lr}_loss_record.csv','w') as f:
            for i,st in enumerate(status):
                if st:
                    str_st = '1'
                else:
                    str_st = '0'
                f.write(str(i+1)+','+str_st+'\n')


parser = argparse.ArgumentParser()
parser.add_argument('--tag')
tag = parser.parse_args().tag

reslut_path ='/home/zzp/FEC_Test/conference_test_burst_CAFEC'
if not os.path.exists(f'{reslut_path}/network_trace/{tag}'):
    os.mkdir(f'{reslut_path}/network_trace/{tag}')
#loss_simulate(reslut_path,50000,tag)
loss_simulate1(reslut_path,50000,tag)
