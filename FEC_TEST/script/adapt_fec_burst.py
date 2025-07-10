import sys
import random
import pandas as pd

import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import math
import argparse
from collections import Counter


start_code_3 = b'\x00\x00\x01'
start_code_4 = b'\x00\x00\x00\x01'
#[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]




raw_frame_path = '/home/zzp/FEC_Test/raw_frame'

def all_zero(data):
    return all(byte == 0 for byte in data)

def canrecover(data,redundancy):
    count = 0
    miss = 0
    for group in data:
        c = 0
        for m in group:
            if(all_zero(m)):
                c += 1
                miss += 1
        if c > redundancy:
            count += 1
    return count, miss


class ExpGolombDecoder:
    """指数哥伦布编码解码器"""
    def __init__(self, data):
        self.data = data
        self.pos = 0  # 当前字节中的位位置（0-7）
        self.index = 0  # 当前字节索引

    def read_bit(self):
        if self.index >= len(self.data):
            return 0
        bit = (self.data[self.index] >> (7 - self.pos)) & 0x1
        self.pos += 1
        if self.pos >= 8:
            self.pos = 0
            self.index += 1
        return bit

    def read_bits(self, n):
        val = 0
        for _ in range(n):
            val = (val << 1) | self.read_bit()
        return val

    def read_ue(self):
        # 解码无符号指数哥伦布编码
        leading_zeros = 0
        while self.read_bit() == 0:
            leading_zeros += 1
        return (1 << leading_zeros) - 1 + self.read_bits(leading_zeros)

def divide_packets(data,packets_size):
    pack_list = [data[i:i+packets_size] for i in range(0,len(data),packets_size)]
    return pack_list

# 返回[[p1,p2,p3,fec],[p1,p2,p3,fec],]
def encoder(packet_list, packet_num, packet_len ,rs_encoder, redundancy, fec_num):
    allp = 0
    groups = [packet_list[i:i+packet_num] for i in range(0, len(packet_list), packet_num)]
    fec_groups = []
    for group in groups:
        #fec_data = rs_encoder.encode(b''.join(group))[-redundancy:]
        encode_group = group
        for i in range(fec_num):
            encode_group.append(b'\x11'*len(group[0]))
        #encode_group.append(bytes(fec_data))
        fec_groups.append(encode_group)
        allp += len(encode_group)
    return fec_groups,allp
# FEC解码恢复 (核心修复)
def decoder(received_pkts, rs_decoder,packet_num):
    # 注意：WebRTC中使用交织式RS编解码（此处简化）
    buffer = b''
    for group in received_pkts:
        try:
            #recovered = rs_decoder.decode(b''.join(group))[0]
            recovered = b''.join(group[:packet_num])
            # print(type(recovered))
            # print(recovered)
            buffer += bytes(recovered)
        except reedsolo.ReedSolomonError:
            print('fail to decode')
    return buffer
# 模拟丢包
def simulate_loss_p(group, loss_rate=0.005):
    received_group = []
    
    for pkt in group:
        if random.random() > loss_rate:
            received_group.append(pkt)
        else:
            received_group.append(b'\x00'*len(pkt))
    return received_group

def simulate_loss(row_len, fec_num, loss_rate=0.005):
    loss_num = 0
    for i in range(row_len+fec_num):
        if random.random() < loss_rate:
            loss_num += 1
    return loss_num

def get_loss_num():
    # 数字列表
    numbers = np.arange(1,14)
    # 对应的概率，必须和数字列表长度相同，且总和为1
    probabilities = [0.372, 0.175, 0.14, 0.035, 0.035, 0.03, 0.03, 0.03, 0.015, 0.015, 0.015, 0.015, 0.093]
    # 使用 random.choices 选择一个数字，k=1 表示选择一个
    selected_number = random.choices(numbers, probabilities, k=1)[0]

    if selected_number==13:
        selected_number = random.randint(13,20)

    return selected_number

def loss_simulate(package_len,loss_rate,loss_future,current_packages, loss_all, reslut_path, fec_num):
    
    pos_2_len ={}
    if loss_future != 0:
        if loss_future < package_len+fec_num:
            pos_2_len[0] = loss_future
        else:
            pos_2_len[0] = package_len+fec_num
    with open(f'{reslut_path}/network_trace/{loss_rate}_loss_record.csv','a') as f:
        for p in range(package_len+fec_num):
            current_packages += 1
            if loss_future == 0:
                if random.random() < loss_rate:
                    loss_future = get_loss_num()
                    if loss_all < current_packages * loss_rate:
                        if loss_future <= package_len+fec_num - p:
                            pos_2_len[p] = loss_future
                        else:
                            pos_2_len[p] = package_len+fec_num - p
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
            
            #f.write(str(package_len)+'\t'+str(fec_num)+'\t'+str(loss_future)+'\t'+str(current_packages)+'\t'+str(pos_2_len)+'\t'+str(loss_all)+'\n')
    return loss_future, current_packages, pos_2_len, loss_all
def exec_loss_I(packet_list, fec_num, pos_2_len,reslut_path):
    loss_count = 0
    total_sum = sum(pos_2_len.values())
    pos_list = []
    if total_sum > fec_num:
        for k,v in pos_2_len.items():
            for i in range(k,v+k):
                if i < len(packet_list):
                    packet_list[i] = b'\x00'*len(packet_list[i])
                    loss_count += 1
                    pos_list.append(i)
    with open(f'{reslut_path}/not_repair_num.csv','a') as f:
        f.write(str(len(packet_list))+','+str(loss_count)+','+str(pos_list)+'\n')
        
            
def exec_loss_P(packet_list, fec_num, fec_num1 ,pos_2_len,reslut_path, front):
    pack_len = len(packet_list)
    total_sum = 0
    total_sum1 = 0
    loss_count = 0
    for k,v in pos_2_len.items():
        for i in range(k,v+k):
            if i < front or (i >= pack_len and i < pack_len + fec_num):
                total_sum+=1
            elif (i >= front and i < pack_len) or (i >= pack_len + fec_num and i < pack_len+fec_num+fec_num1):
                total_sum1+=1

    pos_list = []
    for k,v in pos_2_len.items():
        for i in range(k,v+k):
            if i < front and total_sum > fec_num:
                packet_list[i] = b'\x00'*len(packet_list[i])
                loss_count+=1
                pos_list.append(i)
            elif i >= front and i < len(packet_list) and  total_sum1 > fec_num1:
                packet_list[i] = b'\x00'*len(packet_list[i])
                loss_count+=1
                pos_list.append(front)
    
    with open(f'{reslut_path}/not_repair_num.csv','a') as f:
        f.write(str(len(packet_list))+','+str(loss_count)+','+str(pos_list)+'\n')

def network_trace_load(lr,tag,l):
    loss_dic = {}
    file_path = f'/home/zzp/FEC_Test/conference_test_burst_CAFEC/network_trace/{tag}/{lr}_loss_record.csv'
    raw_data = pd.read_csv(file_path,header=None)
    idx = raw_data[0].tolist()
    sta = raw_data[1].tolist()
    for i,s in zip(idx,sta):
        loss_dic[i] = int(s)
    len_list = []
    for i in range(len(sta)-l):
        count = 0
        for j in range(i,i+l):
            if sta[j] == 1:
                count += 1
        len_list.append(count)
    return loss_dic,len_list
def num_analysis(l):
    count = Counter(l)
    total = len(l)
    all_numbers = sorted(set(l))
    #print(all_numbers)
    counts = [count.get(num, 0) for num in all_numbers]
    percentages = [(count/total)*100 if total > 0 else 0 for count in counts]
    result = []
    current_sum = 0
    for num in percentages:
        current_sum += num
        current_sum = round(current_sum,2)
        result.append(current_sum)
    return result
def decision_maker(p_list,redundary):
    rr = 0.0
    p_ration = 1.0
    rr_len = math.ceil(len(p_list)/1.5)
    print(p_list)
    # for i,p in enumerate(p_list):
    #     if p < 99:
    #         rr_len = i+1
    rr = round(rr_len/ 20,2)
    if p_list[math.ceil(rr_len / 2)] < 60:
        p_ration = 0.3
    if rr<0.01:
        rr = redundary
    return rr,p_ration
    
    
def exc_loss(pack_all,loss_dic,packet_list,fec_num, fec_num1, front, reslut_path,tag):
    pack_len = len(packet_list)
    total_sum = 0
    total_sum1 = 0
    pos_list = []
    for i in range(pack_len+fec_num+fec_num1):
        if loss_dic[pack_all+i+1]==1:
            if (i < front  or (i >= pack_len and i < pack_len + fec_num)):
                    total_sum+=1
            elif (i >= front and i < pack_len) or (i >= pack_len + fec_num and i < pack_len+fec_num+fec_num1):
                    total_sum1+=1
    for i in range(pack_len+fec_num+fec_num1):
        if loss_dic[pack_all+i+1]==1:
            if i < front and total_sum > fec_num:
                packet_list[i] = b'\x00'*len(packet_list[i])
                pos_list.append(i)
            elif i >= front and i < len(packet_list) and  total_sum1 > fec_num1:
                packet_list[i] = b'\x00'*len(packet_list[i])
                pos_list.append(i)
        
    with open(f'{reslut_path}/sim_result/{tag}/not_repair_num.csv','a') as f:
        f.write(str(pack_all)+','+str(len(packet_list))+','+str(pos_list)+'\n')
        
def parse_slice_header(nalu_payload):
    """解析Slice头获取slice_type"""
    if len(nalu_payload) < 1:
        return None

    decoder = ExpGolombDecoder(nalu_payload)
    try:
        first_mb_in_slice = decoder.read_ue()
        slice_type = decoder.read_ue()
        return slice_type
    except:
        return None

def classify_nalu(nalu_type, nalu_payload):
    """根据NALU类型和Slice头分类帧类型"""
    if nalu_type == 5:
        return 'IDR'  # IDR帧

    elif nalu_type == 1:
        slice_type = parse_slice_header(nalu_payload)
        if slice_type is None:
            return 'Unknown'
        # 根据slice_type判断
        if slice_type in [2, 4, 7, 9]:
            return 'I'
        elif slice_type in [1, 6]:
            return 'B'
        elif slice_type in [0, 5]:
            return 'P'
        else:
            return 'Unknown'
    elif nalu_type== 6:
        return 'SEI'  #辅助信息
    elif nalu_type== 7:
        return 'SPS'  #序列参数集
    elif nalu_type== 8:
        return 'PPS'  #图像参数集
    elif nalu_type== 9:
        return 'AUD'  #访问单元分隔符

def split_nalus(h264_data):
    """分割原始数据为NALU单元（含起始码）"""

    nalu_list = []
    buffer = bytearray()

    i = 0
    while i < len(h264_data):
        # 检测3字节起始码
        if i+3 <= len(h264_data) and h264_data[i:i+3] == start_code_3:
            if len(buffer) > 3:
                nalu_list.append(bytes(buffer))
                buffer = bytearray()
            for c in range(3):
                buffer.append(h264_data[i])
                i += 1
        # 检测4字节起始码
        elif i+4 <= len(h264_data) and h264_data[i:i+4] == start_code_4:
            if len(buffer) > 4:
                nalu_list.append(bytes(buffer))
                buffer = bytearray()
            for c in range(4):
                buffer.append(h264_data[i])
                i += 1
        else:
            buffer.append(h264_data[i])
            i += 1

    if len(buffer) > 0:
        nalu_list.append(bytes(buffer))
    return nalu_list

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--thr')
    parser.add_argument('--lr')
    parser.add_argument('--muti_rr',default=2.0)
    parser.add_argument('--muti_pace',default=0.1)
    parser.add_argument('--count_pace',default=1)
    parser.add_argument('--p_ratio',default=0.3)
    parser.add_argument('--bottom',default=0.5)
    parser.add_argument('--divide',default=1)
    return parser.parse_args()
pace_len = 100
threshlod = 1.0


def process_h264_file(input_path):
    """主处理函数"""
    args = arg_parse()
    divd = int(args.divide)
    reslut_path = args.path #'/home/zzp/FEC_Test/conference_test_burst_sFEC'
    
    threshlod = float(args.thr)
    muti_lr = float(args.lr)
    muti_rr = round(float(args.muti_rr),1)
    muti_pace = round(float(args.muti_pace),1)
    count_pace = int(args.count_pace)
    p_ratio = round(float(args.p_ratio),1)
    bottom = round(float(args.bottom),1)
    tag = f'muti_lr_{muti_lr}_muti_rr_{muti_rr}_muti_pace_{muti_pace}_count_pace_{count_pace}_p_ratio_{p_ratio}_bottom_{bottom}'
    loss_rate = np.linspace(0.01,0.1,10)
    red_rate = []
    if not os.path.exists(f'{reslut_path}/sim_result/{tag}'):
        os.mkdir(f'{reslut_path}/sim_result/{tag}')
    if muti_lr > 1:
        red_rate = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
        red_rate = [round(los * muti_lr,3) for los in red_rate]
        #red_rate = [0.3,0.35,0.4,0.45,0.5,0.6,0.65,0.7,0.75,0.8]
    else:
        red_rate = [0.3,0.35,0.4,0.45,0.5,0.6,0.65,0.7,0.75,0.8]
    with open(input_path, 'rb') as f:
        raw_data = f.read()
    ssimedata =pd.read_csv('/home/zzp/FEC_Test/script/ssim_conference.csv',header=None)
    ssim_value = list(ssimedata[0])
    nalus = split_nalus(raw_data)

    for lr,rr in zip(loss_rate,red_rate):
        lr = round(lr,3)
        rr = round(rr,3)
        loss_dic, len_list = network_trace_load(lr,tag,20)
        
        if divd == 1 and lr < 0.06:
            #print('divide is working !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            muti_rr = round(1.2 / rr,2)
            #count_pace = rr*24/1.2
            muti_pace = round((muti_rr-0.5)/10,2)
        with open(f'{reslut_path}/lossrate_{lr}_redundancy_{rr}.h264','wb') as wf:
            CAFEC_count = pace_len
            last_status = False
            buffer = b'' #每个帧的缓存
            received_buffer = b'' #解码后所有的缓存
            count = 0
            I_num = 0
            I_size = 0
            I_code_size = 0
            P_num = 0
            P_size = 0
            P_code_size = 0

            muti_f = 2
            
            redundancy = 100
            packets_num = 20
            packet_size = 1400
            fec_num = int(rr * packets_num)
            
            current_packages = 0
            loss_future = 0
            loss_all = 0
            index = 0
            #with open('sequence.csv','w') as sf:
            for i,nalu in enumerate(nalus):
                if len(nalu) < 1:
                    continue
                # 解析NALU头（第1字节）
                

                nh_nalu =None
                if(nalu.startswith(start_code_3)):
                    nh_nalu = nalu[3:]
                else:
                    nh_nalu = nalu[4:]

                forbidden_bit = (nh_nalu[0] >> 7) & 0x1
                nri = (nh_nalu[0] >> 5) & 0x3
                nal_type = nh_nalu[0] & 0x1F
                
            
                # 分类帧类型
                frame_type = classify_nalu(nal_type, nh_nalu[1:])

                buffer = nalu

                if frame_type == 'IDR' or frame_type == 'I':
                    ssim = float(ssim_value[index])
                    # if current_packages > 1000:
                    #     p_list = num_analysis(len_list[current_packages-1000:current_packages-20])
                    #     rr,p_ratio = decision_maker(p_list=p_list,redundary=rr)
                    index += 1
                    packet_list = divide_packets(buffer,packet_size)
                    I_size += len(packet_list)
                    I_num += 1
                    fec_num = 0
                    fec_num1 = 0
                    # if divd == 1 and lr < 0.06:
                    #     fec_num = math.ceil(1.5*rr * len(packet_list))
                    # else:
                    fec_num = math.ceil(rr * len(packet_list))
                    
                    
                    #loss_future, current_packages, pos_2_len, loss_all = loss_simulate(len(packet_list), lr, loss_future, current_packages, loss_all, reslut_path,fec_num)
                    #exec_loss_I(packet_list,fec_num,pos_2_len,reslut_path) 
                    exc_loss(current_packages,loss_dic,packet_list,fec_num,fec_num1,len(packet_list),reslut_path, tag)
                    I_code_size += (len(packet_list)+fec_num+fec_num1)
                    current_packages += (len(packet_list)+fec_num+fec_num1)
                    wf.write(b''.join(packet_list))
                    # last_status = False
                    # CAFEC_count = pace_len
                    count = 0
                    muti_f = muti_rr
                elif frame_type == 'P':
                    ssim = float(ssim_value[index])
                    index += 1
                    fec_num1 = 0
                    count += 1
                    
                    packet_list = divide_packets(buffer,packet_size)
                    P_size += len(packet_list)
                    P_num += 1
                    front = len(packet_list)

                    if threshlod == 1.0:
                        #print(threshlod,reslut_path)
                        fec_num = math.ceil(rr * len(packet_list))
                    elif ssim < threshlod:
                        #print(ssim,threshlod,reslut_path)
                        cof =1.0
                        if muti_f > 1.0:
                            cof = muti_f
                        fec_num = math.ceil(cof*rr * len(packet_list))
                    else:
                        #print(ssim,threshlod,reslut_path)
                        if muti_f <1.0:
                            front = math.ceil(len(packet_list) * p_ratio)
                            fec_num = math.ceil(muti_f* rr * len(packet_list))
                        else:
                            fec_num = math.ceil(rr * front)
                        #fec_num2 = int(0.5* rr * (len(packet_list)-front))
                    exc_loss(current_packages,loss_dic,packet_list,fec_num,fec_num1,front,reslut_path, tag)
                    current_packages += (len(packet_list)+fec_num+fec_num1)
                    #loss_future, current_packages, pos_2_len, loss_all = loss_simulate(len(packet_list), lr, loss_future, current_packages, loss_all, reslut_path, fec_num+fec_num2)
                    # if ssim < threshlod:
                    #     exec_loss_I(packet_list,fec_num,pos_2_len,reslut_path)
                    # else:
                    #     exec_loss_P(packet_list,fec_num,fec_num2,pos_2_len,reslut_path,front)
                    wf.write(b''.join(packet_list))
                    P_code_size += (len(packet_list) + fec_num + fec_num1)
                    if count %count_pace == 0 and muti_f>bottom:
                        muti_f -= muti_pace
                else:
                    wf.write(buffer)                    
        with open(f'{reslut_path}/sim_result/{tag}/video_size.csv','a') as f:
            f.write(f'{str(lr)},{str(rr)},{str(I_code_size +P_code_size)}\n')

if __name__ == "__main__":
    #/home/zzp/FEC_Test/360P_400k.h264
    #/home/zzp/FEC_Test/480P_800k.h264
    #/home/zzp/FEC_Test/540P_1200k.h264
    #/home/zzp/FEC_Test/720P_2400k.h264
    #/home/zzp/FEC_Test/conference_test_rr=10lr/conference_test.h264
    #/home/zzp/FEC_Test/conference_test_burst_test/raw_video/raw_test_video.h264
    process_h264_file("/home/zzp/FEC_Test/conference_test_burst_test/raw_video/raw_test_video.h264")