#CUDA_VISIBLE_DEVICES=4,5,6,7 eval/proactive/qvh/run_qvh.sh

#!/usr/bin/env bash
set -euo pipefail

########################################
#        Configuration (modify as needed)        #
########################################

MODEL_PATH="whole_model/model"
HEAD_TYPE="mlp"
BASE_OUTPUT_PREFIX="eval/proactive/qvh/results/test"

############################

TEST_DATA_PATH="qvh_val_proactive_tts_merged_silent.jsonl"
PYTHON_BIN="xxxx/venv/llamafactory/bin/python3.10"
SCRIPT_PATH="eval/proactive/omnimmi/omnimmi_qvh.py"

########################################
#      Parse CUDA_VISIBLE_DEVICES      #
########################################

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Default GPU 0"
  export CUDA_VISIBLE_DEVICES=0
fi

IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_SHARDS=${#DEVICES[@]}

echo "Detected ${NUM_SHARDS} GPUs: ${DEVICES[*]}"
echo "The test dataset will be evenly split into ${NUM_SHARDS} shards, one per GPU."

########################################
#   trap: clean up child processes on Ctrl+C   #
########################################

PIDS=()

cleanup() {
  echo "Interrupt signal received, cleaning up child processes..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Killing child process PID=${pid}"
      kill "$pid" 2>/dev/null || true
    fi
  done
  exit 1
}

trap cleanup INT TERM

########################################
#        Launch shard processes        #
########################################

for i in "${!DEVICES[@]}"; do
  DEV=${DEVICES[$i]}
  OUT_FILE="${BASE_OUTPUT_PREFIX}.shard${i}.jsonl"

  echo "Launching shard ${i}/${NUM_SHARDS}, using GPU ${DEV}, output -> ${OUT_FILE}"

  CUDA_VISIBLE_DEVICES=$DEV \
  "${PYTHON_BIN}" "${SCRIPT_PATH}" \
    --model_path "${MODEL_PATH}" \
    --test_data_path "${TEST_DATA_PATH}" \
    --score_output_path "${OUT_FILE}" \
    --head "${HEAD_TYPE}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_id "${i}" &

  PIDS+=($!)
done

echo "All shards launched, waiting for completion..."
wait

echo "All shards finished, starting result merge..."

########################################
#           Merge shard results        #
########################################

MERGED_OUTPUT="${BASE_OUTPUT_PREFIX}.jsonl"
cat "${BASE_OUTPUT_PREFIX}".shard*.jsonl > "${MERGED_OUTPUT}"

echo "Merge completed: ${MERGED_OUTPUT}"
echo "Deleting intermediate shard files..."

# Automatically delete intermediate files
rm -f "${BASE_OUTPUT_PREFIX}".shard*.jsonl || true

echo "Intermediate files deleted, only keeping: ${MERGED_OUTPUT}"