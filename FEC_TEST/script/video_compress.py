import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

video_path = '/home/zzp/youtubeucgdateset'
compress_path = '/home/zzp/FEC_Test/raw_stream'
file_list = os.listdir(video_path)

    

    
def encode_2B(video_path, save_path):
    subprocess.run(f'ffmpeg -i {video_path} -c:v libx264  -g 30 -preset ultrafast -crf 23 -bf 2 {save_path}',shell=True)
    
def encode_23_5000KB(video_path, save_path):
    subprocess.run(f'ffmpeg -i {video_path} -c:v libx264  -g 30 -preset ultrafast -crf 23 -maxrate 5000k -bufsize 10000k {save_path}',shell=True)
def encode_5000KB(video_path, save_path):
    subprocess.run(f'ffmpeg -i {video_path} -c:v libx264  -g 30 -preset ultrafast -b:v 5000k -maxrate 5000k -bufsize 10000k {save_path}',shell=True)
# Use ProcessPoolExecutor to parallelize the process    
def encode_preset(video_path, save_path):
    subprocess.run(f'ffmpeg -i {video_path} -c:v libx264  -g 30 -preset veryfast -crf 23 -maxrate 5000k -bufsize 10000k {save_path}',shell=True)
    
    
if __name__ == '__main__':
    video_path_list = []
    encode_path_list = []
    encode_path_list1 = []
    encode_path_list2 = []
    for file in file_list:
        if file.endswith('.mp4'):
            input_file = os.path.join(video_path, file)
            video_path_list.append(input_file)
            
            output_file = os.path.join(compress_path+'_2B', file.split('.')[0] + '.h264')
            encode_path_list.append(output_file)
            
            output_file1 = os.path.join(compress_path+'_5000K', file.split('.')[0] + '.h264')
            encode_path_list1.append(output_file1)
            
            output_file2 = os.path.join(compress_path+'_preset', file.split('.')[0] + '.h264')
            encode_path_list2.append(output_file2)
            
    # with ProcessPoolExecutor(max_workers=18) as executor:
    #     executor.map(siti, video_path_list, siti_path_list)
    # with ProcessPoolExecutor(max_workers=18) as executor:
    #     executor.map(encode_2B, video_path_list, encode_path_list)
    # with ProcessPoolExecutor(max_workers=18) as executor:
    #     executor.map(encode_5000KB, video_path_list, encode_path_list1)
    with ProcessPoolExecutor(max_workers=18) as executor:
        executor.map(encode_preset, video_path_list, encode_path_list2)
            