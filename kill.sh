#!/bin/bash

# 您想要操作的GPU ID列表 (0-indexed)
TARGET_GPUS=(4 5 6 7)

echo "警告：此脚本将尝试使用 'kill -9' 强制终止在指定GPU（ID: ${TARGET_GPUS[*]}）上的所有进程。"
echo "这可能导致相关程序数据丢失。"
read -p "您确定要继续吗? (请输入 'yes' 继续): " CONFIRMATION

if [ "$CONFIRMATION" != "yes" ]; then
    echo "操作已取消。"
    exit 0
fi

ALL_PIDS_TO_KILL=""

for GPU_ID_TO_KILL in "${TARGET_GPUS[@]}"; do
    echo ""
    echo "正在查找 GPU ${GPU_ID_TO_KILL} 上的进程..."

    # 获取在特定GPU上运行的进程PID列表
    # AWK脚本解释:
    # - in_processes_section: 标志位，表示是否进入了"Processes:"段落
    # - /Processes:/: 匹配到"Processes:"行，设置标志位，并使用getline跳过后续3行表头
    #                 (GPU/PID表头, ID表头, === 分隔线)
    # - 如果在"Processes:"段落内:
    #   - $1 == "|": 确保是进程信息行 (以"|"开头)
    #   - $2 == gpu_target: 确保是目标GPU的行 ($2 是GPU ID列)
    #   - $5 ~ /^[0-9]+$/: 确保PID字段($5)是数字 (PID是第5个有效字段，在有GI/CI ID时)
    #   - print $5: 打印PID
    #   - /^\+--/: 如果匹配到表格结束行 "+--...", 则认为"Processes:"段落结束
    CURRENT_GPU_PIDS=$(nvidia-smi | awk -v gpu_target="${GPU_ID_TO_KILL}" '
        BEGIN { in_processes_section=0 }
        /Processes:/ {
            in_processes_section=1
            # Skip the 3 header lines within the Processes section
            getline
            getline
            getline
            next
        }
        {
            if (in_processes_section) {
                # Process lines look like:
                # "|    0   N/A  N/A      1234      C   /usr/bin/python                   100MiB |"
                # $1 is "|", $2 is GPU_ID, $3 is GI, $4 is CI, $5 is PID
                if ($1 == "|" && $2 == gpu_target && $5 ~ /^[0-9]+$/) {
                    print $5
                }
                # If we encounter another "+----...----+" line, it means the end of the process list for nvidia-smi
                if ($1 ~ /^\+--/) {
                    in_processes_section=0 # Stop processing after this section
                }
            }
        }
    ')

    if [ -n "${CURRENT_GPU_PIDS}" ]; then
        echo "GPU ${GPU_ID_TO_KILL}: 找到以下 PIDs 将被添加到终止列表:"
        echo "${CURRENT_GPU_PIDS}"
        ALL_PIDS_TO_KILL="${ALL_PIDS_TO_KILL} ${CURRENT_GPU_PIDS}"
    else
        echo "GPU ${GPU_ID_TO_KILL}: 未找到正在运行的进程。"
    fi
done

if [ -n "${ALL_PIDS_TO_KILL}" ]; then
    # 去重PIDs，并确保PID是有效的数字
    UNIQUE_PIDS=$(echo "${ALL_PIDS_TO_KILL}" | tr ' ' '\n' | grep '^[0-9]\+$' | sort -u | tr '\n' ' ')

    if [ -n "${UNIQUE_PIDS}" ]; then
        echo ""
        echo "将要尝试终止的PIDs (去重后): ${UNIQUE_PIDS}"
        for PID_TO_KILL in $UNIQUE_PIDS; do
            echo "正在终止 PID: ${PID_TO_KILL} ..."
            sudo kill -9 "${PID_TO_KILL}" # 使用sudo确保权限，如果不需要或者不应使用sudo，请移除
            if [ $? -eq 0 ]; then
                echo "PID ${PID_TO_KILL} 的终止信号已成功发送。"
            else
                echo "警告: 无法向 PID ${PID_TO_KILL} 发送终止信号。可能进程已不存在或权限不足（即使有sudo）。"
            fi
        done
        echo ""
        echo "所有选定GPU上的进程终止操作已尝试完毕。"
    else
        echo ""
        echo "在筛选和去重后，没有有效的PID需要终止。"
    fi
else
    echo ""
    echo "在所有指定的GPU (${TARGET_GPUS[*]}) 上均未找到需要终止的进程。"
fi

echo "脚本执行完成。"