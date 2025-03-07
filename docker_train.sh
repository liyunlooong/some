#!/bin/bash
# docker_train.sh
#
# Usage: bash docker_train.sh <path_to_dataset> <path_to_output> [additional train.py flags]
#
# This script mounts the dataset and output directories to /data/dataset and /data/output
# inside the Docker container and then runs:
#   python train.py -s /data/dataset -m /data/output <extra flags>

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <path_to_dataset> <path_to_output> [additional train.py flags]"
  exit 1
fi

DATASET_DIR="$1"
OUTPUT_DIR="$2"
PORT="${3:-6009}"
IP="${4:-127.0.0.1}"
shift 4

docker run --rm --gpus all \
  -v /tmp/NVIDIA:/tmp/NVIDIA \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -v "$DATASET_DIR":/data/dataset \
  -v "$OUTPUT_DIR":/data/output \
  -v "$(pwd)":/ever_training2 \
  -p "$IP:$PORT:$PORT" \
  halfpotato/ever:latest \
  bash -c 'source activate ever && cd /ever_training2 && rm -r ever && cp -r /ever_training/ever . && python train.py -s /data/dataset -m /data/output "$@"' _ "$@"

