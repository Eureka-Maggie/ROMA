import yaml
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch 
# import wandb
# wandb.init(project="test", name="hello-world", notes="labmda")

try:
    import llamafactory
except ImportError:
    project_root = os.path.abspath(os.path.dirname(__file__))
    src_path = os.path.join(project_root, "src")
    if os.path.isdir(src_path):
        print(f"Adding LLaMA-Factory src directory to sys.path: {src_path}")
        sys.path.insert(0, src_path)
    else:
        print(f"Warning: LLaMA-Factory src directory not found at {src_path}. "
              "Ensure LLaMA-Factory is installed or PYTHONPATH is set correctly.")

from llamafactory.train.tuner import run_exp

def main_debug_exp(yaml_config_path: str):
    """
    Loads configuration from a YAML file and calls run_exp for single-GPU debugging.
    """
    print(f"Loading configuration from: {yaml_config_path}")
    with open(yaml_config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # print("--- Initial Configuration from YAML ---")
    # for key, value in config_dict.items():
    #     print(f"{key}: {value}")
    # print("--------------------------------------")

    # --- Prepare for Single-GPU, Non-Distributed Execution ---
    # config_dict["local_rank"] = -1

    
    if "use_ray" in config_dict and config_dict["use_ray"]:
        print("Warning: 'use_ray' is true in YAML. Forcing to False for single-GPU direct debug.")
        config_dict["use_ray"] = False
    
    if "deepspeed" in config_dict and config_dict["deepspeed"]:
        print(f"Warning: 'deepspeed' found in YAML ({config_dict['deepspeed']}). Forcing to None for single-GPU debug.")
        config_dict["deepspeed"] = None

    callbacks_list = None

    run_exp(
        args=config_dict,
        callbacks=callbacks_list
    )

    print("run_exp finished or debugger detached.")

if __name__ == "__main__":
    yaml_file_path = "yamls/train_ds.yaml"
    
    if not os.path.exists(yaml_file_path):
        print(f"Error: YAML configuration file not found at '{yaml_file_path}'")
        print("Please update the 'yaml_file_path' variable in this script.")
        sys.exit(1)
    
    # Optional: Pass YAML path as a command-line argument
    # if len(sys.argv) > 1:
    #     yaml_file_path_cli = sys.argv[1]
    #     if os.path.exists(yaml_file_path_cli):
    #         yaml_file_path = yaml_file_path_cli
    #     else:
    #         print(f"Warning: YAML path from CLI '{yaml_file_path_cli}' not found. Using hardcoded path.")

    main_debug_exp(yaml_file_path)