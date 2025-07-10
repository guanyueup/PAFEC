import subprocess
import numpy as np

# fps = 24
mr_list = [1.5]#,2.5]#,3.0,3.5,4.0]
mp_list = [1.0]#,0.15]#,0.2,0.25,0.3]
ml = [2.0,3.0,4.0,5.0,6.0,7.0,8.0]
# bt_list = [0.9,0.8,0.7,0.6,0.5]
# ct_list = []
for m in ml:
    for mr,mp in zip(mr_list,mp_list):
        muti_lr = m
        muti_rr = round(mr,1)
        muti_pace = round(mp,2)
        count_pace = 12
        p_ratio = round(1.0,1)
        bottom = round(0.5,1)
        tag = f'muti_lr_{muti_lr}_muti_rr_{muti_rr}_muti_pace_{muti_pace}_count_pace_{count_pace}_p_ratio_{p_ratio}_bottom_{bottom}'
        for i in range(30):
            subprocess.run(f'python3 /home/zzp/FEC_Test/script/loss_simulate.py --tag {tag}',shell=True)
            subprocess.run(f'python3 /home/zzp/FEC_Test/script/adapt_fec_burst.py --path /home/zzp/FEC_Test/conference_test_burst_CAFEC --thr 0.95 --lr {muti_lr} --muti_rr {muti_rr} --muti_pace {muti_pace} --count_pace {count_pace} --p_ratio {p_ratio} --bottom {bottom} --divide 0',shell=True)
            subprocess.run(f'python3 /home/zzp/FEC_Test/script/adapt_fec_burst.py --path /home/zzp/FEC_Test/conference_test_burst_sFEC --thr 1.0 --lr {muti_lr} --muti_rr {muti_rr} --muti_pace {muti_pace} --count_pace {count_pace} --p_ratio {p_ratio} --bottom {bottom} --divide 0',shell=True)
            subprocess.run('/home/zzp/FEC_Test/conference_test_burst_CAFEC/h264tomp4.sh',shell=True)
            subprocess.run('/home/zzp/FEC_Test/conference_test_burst_sFEC/h264tomp4.sh',shell=True)
            subprocess.run(f'python3 /home/zzp/FEC_Test/script/get_ssim_psnr.py --path /home/zzp/FEC_Test/conference_test_burst_CAFEC --lr {muti_lr} --tag {tag}',shell=True)
            subprocess.run(f'python3 /home/zzp/FEC_Test/script/get_ssim_psnr.py --path /home/zzp/FEC_Test/conference_test_burst_sFEC --lr {muti_lr} --tag {tag}',shell=True)
            subprocess.run('/home/zzp/FEC_Test/conference_test_burst_CAFEC/remove_temp.sh',shell=True)
            subprocess.run('/home/zzp/FEC_Test/conference_test_burst_sFEC/remove_temp.sh',shell=True)