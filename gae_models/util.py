# Utility methods for dealing with the autoencoders

import torch
import warnings
import subprocess
import json
import random
import numpy as np

from .MIAGAE import MIAGAE
from .SAG_model import SAG_model

from .terminal_colors import tcols

def define_torch_device(dev):

    if dev == "cpu":
        device = torch.device("cpu")
        print("\033[92mUsing device:\033[0m", device)
        return device
    
    # Use gpu for training if available. Alert the user if not and use cpu.
    print("\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        device = torch.device(
            "cuda:" + str(get_free_gpu()) if torch.cuda.is_available() else "cpu"
        )
        if len(w):
            print(tcols.WARNING + "GPU not available." + tcols.ENDC)

    print("\033[92mUsing device:\033[0m", device)
    return device

def get_free_gpu(threshold_vram_usage=3000, max_gpus=1):
    """
    Returns the free gpu numbers on your system, to replace x in the string 'cuda:x'.
    The freeness is determined based on how much memory is currently being used on a
    gpu.

    Args:
        threshold_vram_usage: A GPU is considered free if the vram usage is below the
            threshold.
        max_gpus: Max GPUs is the maximum number of gpus to assign.
    """

    # Get the list of GPUs via nvidia-smi.
    smi_query_result = subprocess.check_output(
        "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
    )
    # Extract the usage information
    gpu_info = smi_query_result.decode("utf-8").split("\n")
    gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
    gpu_info = [int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info]

    # Keep gpus under threshold only.
    free_gpus = [str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage]
    free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
    gpus_to_use = ",".join(free_gpus)

    if not gpus_to_use:
        raise RuntimeError(tcols.FAIL + "No free GPUs found." + tcols.ENDC)

    return gpus_to_use

def choose_gae_model(ae_type, device, hyperparams) -> callable:
    """
    Picks and loads one of the implemented autoencoder model classes.
    @ae_type     :: String of the type of autoencoder that you want to load.
    @device      :: String of the device to load it on: 'cpu' or 'gpu'.
    @hyperparams :: Dictionary of the hyperparameters to load with.

    returns :: The loaded autoencoder model with the given hyperparams.
    """
    switcher = {
        "MIAGAE": lambda: MIAGAE(device=device, hpars=hyperparams).to(device),
        "SAG_model": lambda: SAG_model(device=device, hpars=hyperparams).to(device)
    }
    model = switcher.get(ae_type, lambda: None)()
    if model is None:
        raise TypeError("Specified AE type does not exist!")

    return model

def import_hyperparams(hyperparams_file) -> dict:
    """
    Import hyperparameters of an ae from json file.
    @model_path :: String of the path to a trained pytorch model folder
                   to import hyperparameters from the json file inside
                   that folder.

    returns :: Imported dictionary of hyperparams from .json file inside
        the trained model folder.
    """
    hyperparams_file = open(hyperparams_file)
    hyperparams = json.load(hyperparams_file)
    hyperparams_file.close()

    return hyperparams

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():   
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)