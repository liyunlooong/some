#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(
        description="Run a Docker container for training with dataset & output mounted."
    )
    # Named argument -s / --scene => dataset directory
    parser.add_argument(
        "-s", "--scene",
        required=True,
        help="Path to the dataset directory (e.g. /data/nerf_datasets/zipnerf_ud/london)."
    )
    # Named argument -m / --model_path => output directory
    parser.add_argument(
        "-m", "--model_path",
        default="./model_output",
        help="Path to the model/output directory."
    )
    # Optional port/ip
    parser.add_argument(
        "--port",
        default="6009",
        help="Port to map inside the container. Defaults to 6009."
    )
    parser.add_argument(
        "--ip",
        default="127.0.0.1",
        help="IP to bind the port to. Defaults to 127.0.0.1."
    )

    # Use parse_known_args to capture any extra flags (unknown) 
    # that we want to forward to train.py:
    known_args, unknown_args = parser.parse_known_args()

    # Now build the docker command:
    docker_cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", "/tmp/NVIDIA:/tmp/NVIDIA",
        # "--user", "$(id -u):$(id -g)",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility",
        # Mount the scene/dataset directory and model_path
        "-v", f"{known_args.scene}:/data/dataset",
        "-v", f"{known_args.model_path}:/data/output",
        # Also mount the current directory for your code
        #"-v", f"{os.getcwd()}:/ever_training2",
        # Port mapping
        "-p", f"{known_args.ip}:{known_args.port}:{known_args.port}",
        "halfpotato/ever:latest",
        "bash", "-c",
        (
            "source activate ever && "
            # "cd /ever_training2 && "
            # "rm -r ever && "
            # "cp -r /ever_training/ever . && "
            # "$@" references extra arguments from the final "_" placeholder
            "python train.py -s /data/dataset -m /data/output \"$@\""
        ),
        "_"  # Placeholder for extra arguments
    ]

    # Append the unknown_args so train.py sees them
    docker_cmd += unknown_args
    print(docker_cmd)

    print("Running:", " ".join(docker_cmd))  # For debugging
    subprocess.run(docker_cmd, check=True)

if __name__ == "__main__":
    main()

