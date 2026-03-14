export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL

ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}

export MASTER_ADDR=$METIS_WORKER_0_HOST
export MASTER_PORT=$port
export WORLD_SIZE=$((ARNOLD_WORKER_NUM * ARNOLD_WORKER_GPU))
export RANK=$ARNOLD_ID

sudo apt update && sudo apt install -y ffmpeg

bin/torchrun \
    --nproc_per_node ${ARNOLD_WORKER_GPU} \
    --nnodes ${ARNOLD_WORKER_NUM} \
    --node_rank ${RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    launcher.py \
    yamls/train.yaml