#!/bin/bash

input_dir="/home/zzp/FEC_Test/conference_test_burst_sFEC"
# 检查目录是否存在
if [ ! -d "${input_dir}" ]; then
    echo "错误：目录 ${input_dir} 不存在！"
    exit 1
fi

# # 打印即将删除的文件列表
# echo "以下文件将被删除："
# find "${input_dir}" -type f -name "*.mp4.h264"

# # 提示用户确认删除操作
# read -p "确认删除以上文件吗？(y/n): " confirm
# if [[ "${confirm}" != "y" ]]; then
#     echo "操作已取消。"
#     exit 0
# fi

# 执行删除操作
find "${input_dir}" -type f -name "*.h264" -exec rm -f {} \;
find "${input_dir}" -type f -name "*.mp4" -exec rm -f {} \;

echo "所有 .mp4.h264 文件已删除。"