#!/bin/bash

# 输入文件夹
input_dir="/home/zzp/FEC_Test/conference_test_burst_CAFEC"

# 遍历所有 .h264 文件
for file in "$input_dir"/*.h264
do
    # 取出不带扩展名的文件名
    filename=$(basename "$file" .h264)
    
    # 输出文件路径
    output_file="${input_dir}/${filename}.mp4"

    echo "正在处理: $file -> $output_file"

    # 执行封装命令，自动生成PTS，防止警告
    #ffmpeg -fflags +genpts -r 25 -i -c copy guess_mvs+deblock+favor_inter
    #ffmpeg -fflags +genpts -err_detect ignore_err -r 25 -i "$file"  -c:v copy -bsf:v h264_mp4toannexb "$output_file"
    ffmpeg -fflags +genpts -err_detect ignore_err -ec deblock+favor_inter -r 25 -i "$file" -c:v libx264 -preset ultrafast "$output_file" -y


done

echo "全部处理完毕！"
