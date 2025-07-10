import subprocess

for pos in range(1,10):
    for i in range(1,10):
        subprocess.run(f'python3 adapt_fec_burst——test.py --path /home/zzp/FEC_Test/conference_test_burst_test --pos {pos} --index {i} --len 10',shell=True)
        subprocess.run(f'./h264tomp4.sh',shell=True)
        subprocess.run(f'python3 get_ssim_psnr_test.py --path /home/zzp/FEC_Test/conference_test_burst_test --pos {pos} --index {i} --len 10',shell=True)