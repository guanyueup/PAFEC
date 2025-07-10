from adapt_fec import split_nalus, classify_nalu, parse_slice_header, divide_packets, ExpGolombDecoder
import os
import pandas as pd

start_code_3 = b'\x00\x00\x01'
start_code_4 = b'\x00\x00\x00\x01'

h264_dir = '/home/zzp/FEC_Test/conference_test'
ssim_dir = '/home/zzp/FEC_Test/ssim'
file_list = os.listdir(h264_dir)

def is_csv_empty(file_path):
    return os.path.getsize(file_path) == 0


with open('conf_frame_info.csv', 'w') as f:
    f.write('frame_type,frame_size\n')
    for file in file_list:
        I_num = 0
        P_num = 0
        B_num = 0
        I_size = 0
        P_size = 0
        B_size = 0
        
        if file.endswith('.h264'):
            h264_path = os.path.join(h264_dir, file)
            #ssim_path = os.path.join(ssim_dir, file.split('.')[0]+'.csv')
            
            with open(h264_path, 'rb') as bf:
                raw_data = bf.read()
            # if is_csv_empty(ssim_path):
            #     continue
            #ssim_list = pd.read_csv(ssim_path, header=None)[0].tolist()
            print(f'Processing {h264_path}...')
            nalus = split_nalus(raw_data)
            print(f'NALUs count: {len(nalus)}')
            count = -1
            for i, nalu in enumerate(nalus):
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
 
                if frame_type == 'IDR' or frame_type == 'I' or frame_type == 'P' or frame_type == 'B':
                    # if count == -1:
                    #     ssim = 0.0
                    # else:
                    #     ssim = ssim_list[count]
                    f.write(f'{frame_type},{len(nalu)}\n')
                    count+= 1
            