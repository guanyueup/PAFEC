import sys
from reedsolo import RSCodec
import reedsolo
import random
import pandas as pd

import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os

import argparse


start_code_3 = b'\x00\x00\x01'
start_code_4 = b'\x00\x00\x00\x01'

loss_rate = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
red_rate_all = [0.05 , 0.1,  0.15, 0.2,  0.25, 0.3,  0.35, 0.4,  0.45, 0.5 ]
red_rate = [0.035, 0.07, 0.105, 0.14, 0.175, 0.21, 0.245, 0.28, 0.315, 0.35]

loss_rate1 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]


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
    probabilities = [0.372, 0.175, 0.14, 0.035, 0.035, 0.03, 0.03, 0.03, 0.015, 0.015, 0.015, 0.015, 0.102]
    # 使用 random.choices 选择一个数字，k=1 表示选择一个
    selected_number = random.choices(numbers, probabilities, k=1)[0]

    if selected_number==13:
        selected_number = random.randint(13,125)

    return selected_number

def loss_simulate(package_len,loss_rate,loss_future,current_packages, loss_all, reslut_path, fec_num):
    
    pos_2_len ={}
    if loss_future != 0:
        pos_2_len[0] = loss_future
    with open(f'{reslut_path}/loss_record.txt','a') as f:
        for p in range(package_len+fec_num):
            if loss_future == 0:
                if random.random() < loss_rate:
                    loss_future = get_loss_num()
                    if loss_all < current_packages * loss_rate and (loss_all + loss_future) < (3017*(1+7*loss_rate))*loss_rate:
                        pos_2_len[p] = loss_future
                        loss_future -= 1
                        loss_all += 1
                    else:
                        loss_future = 0
            else:
                loss_future -= 1
                loss_all += 1
            current_packages += 1
            f.write(str(loss_future)+'\t'+str(current_packages)+'\t'+str(pos_2_len)+'\t'+str(loss_all)+'\n')
    return loss_future, current_packages, pos_2_len, loss_all
def exec_loss(packet_list, fec_num, pos_2_len):
    total_sum = sum(pos_2_len.values())
    count = 0
    if total_sum > fec_num:
        for k,v in pos_2_len.items():
            for i in range(k,v+k):
                if i < len(packet_list):
                    packet_list[i] = b'\x00'*len(packet_list[i])
                    count += 1
    if count < int(len(packet_list)*0.2):
        return False
    return True

def pos_loss(packet_list, pos, loss_len):
    loss_count = 0
    for i in range(pos,pos+loss_len):
        if i <= len(packet_list):
            loss_count += 1
            packet_list[i-1] = b'\x00'*len(packet_list[i-1])
    print(f'pack len{len(packet_list)} pos is {pos} there is loss {loss_count}')

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
    parser.add_argument('--thr',default=1)
    parser.add_argument('--lr',default=1)
    parser.add_argument('--pos',default=-1)
    parser.add_argument('--index',default=-1)
    parser.add_argument('--len',default=-1)
    return parser.parse_args()
pace_len = 100
threshlod = 1


def process_h264_file(input_path):
    """主处理函数"""
    args = arg_parse()
    reslut_path = args.path #'/home/zzp/FEC_Test/conference_test_burst_sFEC'
    threshlod = float(args.thr)
    pos = int(args.pos)
    frame_in = int(args.index)
    loss_len = int(args.len)
    red_rate_list = None
    if int(args.lr) == 1:
        red_rate_list = red_rate_all
    else:
        red_rate_list = red_rate
    
    with open(input_path, 'rb') as f:
        raw_data = f.read()
    ssimedata =pd.read_csv('/home/zzp/FEC_Test/script/ssim_conference.csv',header=None)
    ssim_value = list(ssimedata[0])
    nalus = split_nalus(raw_data)
    for lr,rr in zip([10],[10]):
        with open(f'{reslut_path}/lossrate_{lr}_redundancy_{rr}.h264','wb') as wf:
            CAFEC_count = pace_len
            last_status = False
            buffer = b'' #每个帧的缓存
            received_buffer = b'' #解码后所有的缓存
            count = 0
            IDR_num = 0
            IDR_size = 0
            IDR_code_size = 0
            IDR_fail = 0
            IDR_miss = 0
            I_num = 0
            I_size = 0
            I_code_size = 0
            I_fail = 0
            I_miss = 0
            P_num = 0
            P_size = 0
            P_code_size = 0
            P_fail = 0
            P_miss = 0
            B_num = 0
            B_size = 0
            B_code_size = 0
            B_fail = 0
            B_miss = 0
            
            redundancy = 100
            rs_code = RSCodec(redundancy)

            packets_num = 20
            packet_size = 1400
            fec_num = int(rr * packets_num)
            rround=1
            last_nalu = None

            current_packages = 0
            loss_future = 0
            loss_all = 0
            for c in range(rround):
                index = 0
                #with open('sequence.csv','w') as sf:
                with open(f'{reslut_path}/frame_size.csv','w') as f:
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
                            index += 1
                            packet_list = divide_packets(buffer,packet_size)
                            I_size = len(packet_list)
                            f.write(f'I,{str(I_size)}\n')
                            I_num += 1
                            fec_num = int(rr * len(packet_list))
                            #loss_future, current_packages, pos_2_len, loss_all = loss_simulate(len(packet_list), lr, loss_future, current_packages, loss_all, reslut_path,fec_num)
                            #exec_loss(packet_list,fec_num,pos_2_len)
                            I_code_size += (len(packet_list)+fec_num)
                            wf.write(b''.join(packet_list))
                            last_status = False
                            CAFEC_count = pace_len
                            count = 0
                        elif frame_type == 'P':
                            count += 1
                            ssim = float(ssim_value[index])
                            index += 1
                            
                            packet_list = divide_packets(buffer,packet_size)
                            P_size = len(packet_list)
                            P_num += 1
                            f.write(f'P,{str(P_size)}\n')
                            if ssim < threshlod:
                                fec_num = int(rr * len(packet_list))
                                last_status = False
                                CAFEC_count = pace_len
                            else:
                                if not last_status or CAFEC_count > 0:
                                    fec_num = 0
                                    CAFEC_count -= 1
                                    last_status = True
                                else:
                                    fec_num = int(rr * len(packet_list))
                                    last_status = False
                                    CAFEC_count = pace_len

                            #loss_future, current_packages, pos_2_len, loss_all = loss_simulate(len(packet_list), lr, loss_future, current_packages, loss_all, reslut_path, fec_num)
                            if frame_in == count:
                                pos_loss(packet_list,pos,loss_len)
                            # if use_last and ssim >= threshlod:
                            #     if last_nalu != None:
                            #         wf.write(last_nalu)
                            #         last_nalu = None
                            #     else:
                            #         wf.write(b''.join(packet_list))
                            # else: 
                            last_nalu = b''.join(packet_list)
                            wf.write(last_nalu)
                            P_code_size += (len(packet_list) + fec_num)
                        else:
                            packet_list = divide_packets(buffer,packet_size)
                            wf.write(buffer)                    
        with open(f'{reslut_path}/result.csv','a') as f:
            f.write(f'{str(lr)},{str(rr)},{str(I_code_size/rround +P_code_size/rround)},{str(I_fail/rround)},{str(P_fail/rround)}\n')
        

if __name__ == "__main__":
    #/home/zzp/FEC_Test/360P_400k.h264
    #/home/zzp/FEC_Test/480P_800k.h264
    #/home/zzp/FEC_Test/540P_1200k.h264
    #/home/zzp/FEC_Test/720P_2400k.h264
    #/home/zzp/FEC_Test/conference_test_rr=10lr/conference_test.h264
    #/home/zzp/FEC_Test/conference_test_burst_test/raw_video/raw_test_video.h264
    #/home/zzp/FEC_Test/conference_test_burst_test/more_move_raw_video/raw_test_video.h264
    process_h264_file("/home/zzp/FEC_Test/conference_test_burst_test/raw_video/raw_test_video.h264")